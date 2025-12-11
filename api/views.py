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
from Aiary.celery import app as celery_app
from diary.tasks import chat_reply_task
from accounts.models import Profile

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
    async_result = run_ocr_task.delay(image_path)
    return JsonResponse({"task_id": async_result.id})


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
def chat_history(request):
    session_id = request.GET.get("session_id")
    if not session_id:
        return HttpResponseBadRequest("no session id")

    try:
        session = CounselingSession.objects.get(pk=session_id, user=request.user)
    except CounselingSession.DoesNotExist:
        return HttpResponseBadRequest("invalid session")

    turns = list(
        session.turns.values("sender", "message", "created_at").order_by("created_at")
    )

    entry = session.entry
    emotions = []
    if isinstance(entry.emotion_vector, dict):
        emotions = entry.emotion_vector.get("main_emotions", []) or []

    analysis = {
        "summary": entry.summary or "",
        "suggestions": entry.suggestions or "",
        "emotions": emotions,
        "mood": entry.mood_self_report,
    }

    # 유저별 턴 제한 계산
    profile = getattr(request.user, "profile", None)
    default_max = MAX_TURNS_PER_SESSION
    if profile:
        per_user_max = profile.get_max_turns(default_max)   # None이면 무제한
    else:
        per_user_max = default_max

    user_turn_count = session.turns.filter(sender="user").count()

    return JsonResponse({
        "turns": turns,
        "analysis": analysis,
        "max_turns": per_user_max,  # 5 or None
        "turns_used": user_turn_count,
    })


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

    # 유저 속성 가져오기 (유료인지 무료유저인지)
    user = request.user
    profile = getattr(user, "profile", None)  # Profile 모델
    if profile:
        per_user_max = profile.get_max_turns(MAX_TURNS_PER_SESSION)
    else:
        per_user_max = MAX_TURNS_PER_SESSION

    # 턴 제한 체크
    user_turn_count = session.turns.filter(sender="user").count()
    if per_user_max is not None and user_turn_count >= per_user_max:
        return JsonResponse(
            {
                "error": "turn_limit_reached",
                "max_turns": per_user_max,
                "turns_used": user_turn_count,
                "message": "이번 상담 세션의 무료 메시지 한도를 모두 사용했습니다.",
            }
        )

    # 사용자 턴 저장
    ChatTurn.objects.create(session=session, sender="user", message=message)

    # GPT 답변은 Celery에게 맡김
    chat_reply_task.delay(session.pk)

    # 프론트는 여기서 바로 OK 받고 폴링 시작
    user_turn_count += 1
    return JsonResponse({
        "status": "queued",
        "max_turns": MAX_TURNS_PER_SESSION,
        "turns_used": user_turn_count,
    })

