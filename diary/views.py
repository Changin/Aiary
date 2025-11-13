# diary/views.py
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy, reverse
from django.views.generic import ListView, CreateView, DetailView
from django.shortcuts import redirect
from .models import DiaryEntry, CounselingSession, ChatTurn
from .forms import DiaryEntryForm
from .services import bootstrap_counseling  # 첫 응답 생성 함수


class EntryListView(LoginRequiredMixin, ListView):
    model = DiaryEntry
    template_name = "diary/entry_list.html"
    paginate_by = 10

    def get_queryset(self):
        return DiaryEntry.objects.filter(user=self.request.user)


class EntryCreateView(LoginRequiredMixin, CreateView):
    model = DiaryEntry
    form_class = DiaryEntryForm
    template_name = "diary/entry_form.html"

    def form_valid(self, form):
        form.instance.user = self.request.user
        res = super().form_valid(form)

        # 세션 없으면 생성
        session, created = CounselingSession.objects.get_or_create(
            entry=self.object, user=self.request.user
        )

        # 일기 저장 직후 GPT 호출 + 첫 상담 메시지 생성
        # (프로토타입이라 동기 호출. 나중에 Celery로 비동기 처리 가능)
        try:
            # 턴이 없다면(중복 호출 방지), 부트스트랩 함수 수행 (최초 메세지 턴 생성)
            if not session.turns.exists():
                bootstrap_counseling(self.object, session)
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
        return ctx
