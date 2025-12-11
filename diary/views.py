# diary/views.py
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy, reverse
from django.views.generic import ListView, CreateView, DetailView, DeleteView, TemplateView
from django.shortcuts import redirect
from .models import DiaryEntry, CounselingSession, ChatTurn, DiaryTopic
from .forms import DiaryEntryForm
# from .services import bootstrap_counseling  # 첫 응답 생성 함수
from .tasks import bootstrap_counseling_task    # 비동기 첫 응답 생성 함수


class EntryListView(LoginRequiredMixin, ListView):
    model = DiaryEntry
    template_name = "diary/entry_list.html"
    paginate_by = 10

    def get_queryset(self):
        return DiaryEntry.objects.filter(user=self.request.user).order_by("-entry_date")


class EntryCreateView(LoginRequiredMixin, CreateView):
    model = DiaryEntry
    form_class = DiaryEntryForm
    template_name = "diary/entry_form.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # 랜덤 주제 하나 선택
        topic = (
            DiaryTopic.objects
            .filter(is_active=True)
            .order_by("?")
            .first()
        )
        context["topic_suggestion"] = topic.text if topic else None
        return context

    def form_valid(self, form):
        form.instance.user = self.request.user
        res = super().form_valid(form)

        # 세션 없으면 생성
        session, created = CounselingSession.objects.get_or_create(
            entry=self.object, user=self.request.user
        )

        # 일기 저장 직후 GPT 호출 + 첫 상담 메시지 생성
        # Celery로 비동기 호출 (큐에 작업 적재)
        try:
            # 턴이 없다면(중복 호출 방지), 부트스트랩 함수 수행 (최초 메세지 턴 생성)
            if not session.turns.exists():
                bootstrap_counseling_task.delay(self.object.pk, session.pk)
        except Exception:
            # 오류가 나더라도 일기 저장 자체는 계속 진행
            pass

        # 작성 후 상세(채팅과 함께)로 이동
        return redirect(reverse("diary:detail", args=[self.object.pk]))

    def get_success_url(self):
        return reverse_lazy("diary:detail", args=[self.object.pk])


class EntryDetailView(LoginRequiredMixin, DetailView):
    model = DiaryEntry
    template_name = "diary/entry_detail.html"

    def get_queryset(self):
        # 본인 것만 접근 가능
        return DiaryEntry.objects.filter(user=self.request.user)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        session, _ = CounselingSession.objects.get_or_create(
            entry=self.object, user=self.request.user
        )
        ctx["session"] = session
        # 기존 턴들도 컨텍스트에 추가
        ctx["turns"] = list(
            session.turns.values("sender", "message", "created_at").order_by("created_at")
        )

        # 무료 사용자 턴 제한 (세션당 5개)
        MAX_TURNS = 5
        used_turns = session.turns.filter(sender="user").count()
        ctx["max_turns"] = MAX_TURNS
        ctx["turns_used"] = used_turns
        ctx["turns_remaining"] = max(0, MAX_TURNS - used_turns)

        return ctx


class EntryDeleteView(LoginRequiredMixin, DeleteView):
    model = DiaryEntry
    template_name = "diary/entry_confirm_delete.html"
    success_url = reverse_lazy("diary:list")

    def get_queryset(self):
        # ✅ 본인 일기만 삭제 가능하게 제한
        return DiaryEntry.objects.filter(user=self.request.user)


class EntrySummaryView(LoginRequiredMixin, TemplateView):
    template_name = "diary/entry_summary.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        qs = DiaryEntry.objects.filter(user=self.request.user).order_by("-entry_date", "created_at")

        # 간단한 타임라인용 데이터
        entries = []
        for e in qs:
            emotions = None
            if isinstance(e.emotion_vector, dict):
                emotions = e.emotion_vector.get("main_emotions", [])
            entries.append({
                "id": e.id,
                "date": e.entry_date,
                "title": e.title or "(제목 없음)",
                "mood": e.mood_self_report,
                "emotions": emotions or [],
            })

        ctx["entries"] = entries
        return ctx
