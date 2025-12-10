# api/views.py
import json, os
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from django.conf import settings
from diary.models import CounselingSession, ChatTurn, DiaryEntry
from ocr.tasks import run_ocr_task
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from celery.result import AsyncResult
from Aiary.celery impoort app as celery_app

from openai import OpenAI


client = OpenAI(api_key=settings.SECRETS.get("OPENAI_API_KEY", None))
MODEL_NAME = settings.SECRETS.get("OPENAI_MODEL", "gpt-4.1-mini")

MAX_TURNS_PER_SESSION = 5


@login_required
@require_POST
def ocr_start(request):
    image = request.FILES.get("image")
    if not image:
        return HttpResponseBadRequest("no image")

    # 1) 파일 저장 (MEDIA_ROOT/diary_ocr/...)
    filename = default_storage.save(
        os.path.join("diary_ocr", image.name),
        ContentFile(image.read())
    )
    image_path = os.path.join(settings.MEDIA_ROOT, filename)

    # Celery Task 실행하고 곧바로 task_id 반환
    async_result = run_ocr_task(image_path)
    return JsonResponse({"task_id": async_result})


@login_required
def ocr_result(request):
    task_id = request.GET.get("task_id")
    if not task_id:
        return HttpResponseBadRequest("no task id")

    result = AsyncResult(task_id, app=celery_app)

    if result.state in ("PENDING", "STARTED"):
        return JsonResponse({"status": "pending"})
    elif result.state == "SUCCESS":
        return JsonResponse({
            "status": "success",
            "text": result.result or "",
        })
    elif result.state in ("FAILURE", "REVOKED"):
        return JsonResponse({
            "status": "error",
            "message": "OCR 처리 중 오류가 발생했습니다."
        }, status=500)
    else:
        return JsonResponse({"status": result.state})


@login_required
@require_POST
def chat_send(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
        session_id = int(payload.get("session_id"))
        message = (payload.get("message") or "").strip()
        if not message:
            return HttpResponseBadRequest("empty message")
    except Exception:
        return HttpResponseBadRequest("bad json")

    # 세션 소유권 검증
    try:
        session = CounselingSession.objects.select_related("user", "entry").get(
            pk=session_id, user=request.user
        )
    except CounselingSession.DoesNotExist:
        return HttpResponseBadRequest("invalid session")

    # user 턴 개수 제한 체크
    user_turn_count = session.turns.filter(sender="user").count()
    if user_turn_count >= MAX_TURNS_PER_SESSION:
        return JsonResponse(
            {
                "error": "turn_limit_reached",
                "max_turns": MAX_TURNS_PER_SESSION,
                "turns_used": user_turn_count,
                "message": "이번 상담 세션의 무료 메시지 5개를 모두 사용했습니다.",
            }
        )

    # 사용자 턴 저장
    user_turn = ChatTurn.objects.create(session=session, sender="user", message=message)

    # 현재 일기 내용 가져오기 (컨텍스트에 활용)
    entry: DiaryEntry = session.entry

    # ---- OpenAI GPT 호출 ----
    # 대화 히스토리(최근 n턴만) 구성
    n = 8
    recent_turns = list(
        session.turns.order_by("-created_at")[:n]
    )[::-1]  # 최신 n턴까지, 시간순으로 재정렬

    messages = []

    # 시스템 프롬프트
    system_prompt = settings.COUNSELING_DIALOG_SYSTEM_PROMPT
    messages.append({"role": "system", "content": system_prompt})

    # 일기 요약 컨텍스트
    if entry:
        context_text = f"사용자의 오늘 일기 요약 컨텍스트:\n제목: {entry.title}\n내용: {entry.content}\n"
        if entry.mood_self_report is not None:
            context_text += f"자기 평가 기분 점수(1~10): {entry.mood_self_report}\n"
        messages.append({"role": "system", "content": context_text})

    # 과거 턴들을 messages에 쌓기
    for t in recent_turns:
        role = "assistant" if t.sender == "assistant" else "user"
        messages.append({"role": role, "content": t.message})

    # 마지막에 방금 질문 추가
    messages.append({"role": "user", "content": message})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.6,
            max_tokens=400,
        )
        reply = completion.choices[0].message.content.strip()
    except Exception as e:
        # 에러 시 기본 메시지
        reply = "현재 상담 서버에 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."
        user_turn_count -= 1

    # 어시스턴트 턴 저장
    ChatTurn.objects.create(session=session, sender="assistant", message=reply)

    # 응답에 남은 턴 수 넣어주기
    turns_used_after = user_turn_count + 1
    return JsonResponse(
        {
            "reply": reply,
            "max_turns": MAX_TURNS_PER_SESSION,
            "turns_used": turns_used_after,
        }
    )
