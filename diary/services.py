# diary/services.py
import json
from django.conf import settings
from openai import OpenAI

from .models import DiaryEntry, CounselingSession, ChatTurn

client = OpenAI(api_key=settings.SECRETS.get("OPENAI_API_KEY"))
MODEL_NAME = settings.SECRETS.get("OPENAI_MODEL", "gpt-4.1-mini")


def bootstrap_counseling(entry: DiaryEntry, session: CounselingSession):
    """
    일기가 저장된 직후 호출:
    - GPT로 일기 요약/분석 + 자기 성찰 질문 3개 생성
    - DiaryEntry에 요약/분석 값 저장
    - 첫 번째 assistant ChatTurn 생성
    """

    diary_text = entry.content
    mood = entry.mood_self_report
    date_str = entry.entry_date.strftime("%Y-%m-%d") if entry.entry_date else ""

    system_prompt = (
        "너는 한국어 심리상담 보조자이다. 공감적이고 비판단적이며, "
        "일기에 대한 짧은 요약과 정서 분석, 그리고 일기 내용을 기반으로 자기 성찰을 도와주는 질문을 3개 제공한다. "
        "반말보다는 편안한 존댓말/반존댓말 사이 느낌으로, 부드럽고 따뜻하게 말해라. "
        "자/타해 위험이 강하게 느껴지면 전문 도움을 언급하되 과도하게 자극적 표현은 피한다.\n\n"
        "반드시 JSON 형식으로만 답하라. 추가 텍스트나 설명은 넣지 마라.\n"
        "형식 예시는 다음과 같다:\n"
        "{\n"
        '  \"summary\": \"... 일기 핵심 요약 ...\",\n'
        '  \"main_emotions\": [\"불안\", \"피로\"],\n'
        '  \"advice\": \"... 오늘에 대한 공감 + 조언 ...\",\n'
        '  \"reflection_questions\": [\n'
        '    \"질문1\",\n'
        '    \"질문2\",\n'
        '    \"질문3\"\n'
        "  ]\n"
        "}\n"
    )

    user_content = f"일기 날짜: {date_str}\n"
    if mood is not None:
        user_content += f"사용자가 자기 평가한 기분 점수(1~10): {mood}\n"
    user_content += "\n--- 일기 내용 ---\n"
    user_content += diary_text

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.6,
            max_tokens=600,
        )
        raw = completion.choices[0].message.content
        data = json.loads(raw)
    except Exception:
        # JSON 파싱 실패/호출 오류 시 기본 fallback
        data = {
            "summary": "오늘 일기의 전반적인 맥락은 힘든 감정과 고민이 섞여 있는 하루로 보인다.",
            "main_emotions": ["불안", "피로"],
            "advice": "오늘 스스로를 많이 몰아붙였다면, 지금만큼은 잘 버텨낸 나를 한 번 인정해 주면 좋겠다.",
            "reflection_questions": [
                "오늘 나를 가장 힘들게 했던 한 가지 상황은 무엇이었나요?",
                "그 상황에서 나는 무엇 때문에 그렇게 느꼈던 것 같나요?",
                "내일의 나를 위해 아주 작게라도 해볼 수 있는 변화가 있다면 무엇인가요?",
            ],
        }

    summary = data.get("summary", "").strip()
    advice = data.get("advice", "").strip()
    emotions = data.get("main_emotions") or []
    questions = data.get("reflection_questions") or []

    # DiaryEntry에 요약/분석 값 저장 (일부만 먼저 사용)
    entry.summary = summary
    entry.suggestions = advice
    entry.emotion_vector = {"main_emotions": emotions}
    entry.save(update_fields=["summary", "suggestions", "emotion_vector"])

    # 사용자에게 보여줄 첫 메시지 텍스트 구성
    msg_lines = []
    msg_lines.append("오늘 일기에 대해 Aiary가 정리해본 내용이에요.")
    if summary:
        msg_lines.append("■ 요약")
        msg_lines.append(summary)
        msg_lines.append("")

    if emotions:
        msg_lines.append("■ 지금 마음에서 두드러지는 감정들")
        msg_lines.append(", ".join(emotions))
        msg_lines.append("")

    if advice:
        msg_lines.append("■ 오늘을 위한 작은 제안")
        msg_lines.append(advice)
        msg_lines.append("")

    if questions:
        msg_lines.append("■ 오늘을 돌아보는 자기 성찰 질문 3가지")
        for i, q in enumerate(questions[:3], start=1):
            msg_lines.append(f"{i}. {q}")
        msg_lines.append("")
        msg_lines.append("편하게 하나씩 답해도 좋고, 다른 이야기를 꺼내줘도 괜찮아요.")

    first_message = "\n".join(msg_lines).strip() or (
        "오늘 일기를 잘 읽었어요. 지금 마음 상태를 조금 더 이해해 보고 싶은데, "
        "오늘 나를 가장 힘들게 했던 장면이 있었다면 하나 떠올려볼까요?"
    )

    # 첫 assistant 턴 저장
    ChatTurn.objects.create(
        session=session,
        sender="assistant",
        message=first_message,
    )
