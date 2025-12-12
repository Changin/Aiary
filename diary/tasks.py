# diary/tasks.py
from celery import shared_task
from django.conf import settings
from openai import OpenAI

from .models import DiaryEntry, CounselingSession, ChatTurn
from .services import bootstrap_counseling


client = OpenAI(api_key=settings.OPENAI_API_KEY)
MODEL_NAME = settings.OPENAI_MODEL


@shared_task
def chat_reply_task(session_id: int):
    session = CounselingSession.objects.select_related("entry", "user").get(pk=session_id)
    entry: DiaryEntry = session.entry

    # 메시지 구성 (지금 사용 중인 로직 그대로 옮기면 됨)
    messages = []

    dialog_system_prompt = settings.COUNSELING_DIALOG_SYSTEM_PROMPT
    messages.append({"role": "system", "content": dialog_system_prompt})

    # 일기 컨텍스트
    context_text = f"사용자의 오늘 일기 요약 컨텍스트:\n제목: {entry.title}\n내용: {entry.content}\n"
    if entry.mood_self_report is not None:
        context_text += f"자기 평가 기분 점수(1~10): {entry.mood_self_report}\n"
    if entry.summary:
        context_text += f"\n모델이 분석한 요약: {entry.summary}\n"
    if isinstance(entry.emotion_vector, dict) and entry.emotion_vector.get("main_emotions"):
        context_text += "주요 감정: " + ", ".join(entry.emotion_vector["main_emotions"])
    messages.append({"role": "system", "content": context_text})

    # 과거 턴들
    recent_turns = list(session.turns.order_by("-created_at")[:50])[::-1]
    for t in recent_turns:
        role = "assistant" if t.sender == "assistant" else "user"
        messages.append({"role": role, "content": t.message})

    # 마지막 user 메시지는 이미 ChatTurn으로 저장되어 있다고 가정

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.6,
            max_tokens=400,
        )
        reply = completion.choices[0].message.content.strip()
    except Exception:
        reply = "현재 상담 서버에 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."

    ChatTurn.objects.create(
        session=session,
        sender="assistant",
        message=reply,
    )


@shared_task
def bootstrap_counseling_task(entry_id, session_id):
    entry = DiaryEntry.objects.get(pk=entry_id)
    session = CounselingSession.objects.get(pk=session_id)
    bootstrap_counseling(entry, session)
