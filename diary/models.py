from django.db import models
from django.conf import settings
from django.utils import timezone


# 일기 모델
class DiaryEntry(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="entries"
    )

    # 새로 추가 : 사용자가 선택하는 일기 날짜
    entry_date = models.DateField(default=timezone.localdate)
    title = models.CharField(max_length=120, blank=True)    # 타이틀
    content = models.TextField()   # 프로토타입: 원문 평문 저장
    mood_self_report = models.IntegerField(null=True, blank=True)  # 기분 1~10
    created_at = models.DateTimeField(auto_now_add=True)    # 생성 날짜

    # AI 분석 결과(민감도 낮은 파생 데이터)
    sentiment = models.CharField(max_length=20, blank=True)           # pos/neg/neu 등
    emotion_vector = models.JSONField(default=dict, blank=True)       # {"joy":0.2,...}
    topics = models.JSONField(default=list, blank=True)               # ["학업","관계"]
    summary = models.TextField(blank=True)                            # 짧은 요약
    suggestions = models.TextField(blank=True)                        # 행동/조언

    def __str__(self):
        return f"[{self.created_at:%Y-%m-%d}] {self.title or 'Untitled'}"


# 상담 세션 모델
class CounselingSession(models.Model):
    entry = models.OneToOneField(
        DiaryEntry, on_delete=models.CASCADE,
        related_name="counseling_session",
        null=True, blank=True
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="counseling_sessions"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)  # 세션 종료 여부

    def __str__(self):
        return f"Session {self.pk} for Entry {self.entry_id}"


# 채팅 메세지 모델
class ChatTurn(models.Model):
    session = models.ForeignKey(
        CounselingSession,
        on_delete=models.CASCADE,
        related_name="turns"
    )
    sender = models.CharField(max_length=10)  # "user" 또는 "assistant"
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    # 나중에 토큰 사용량/코스트 기록 가능
    prompt_tokens = models.IntegerField(null=True, blank=True)
    completion_tokens = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.sender}: {self.message[:20]}"
