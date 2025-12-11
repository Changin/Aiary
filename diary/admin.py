from django.contrib import admin
from .models import DiaryEntry, CounselingSession, ChatTurn, DiaryTopic


@admin.register(DiaryTopic)
class DiaryTopicAdmin(admin.ModelAdmin):
    list_display = ("text", "is_active", "created_at")
    list_filter = ("is_active",)
    search_fields = ("text",)


@admin.register(DiaryEntry)
class DiaryEntryAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "entry_date", "title")
    list_filter = ("user", "entry_date")
    search_fields = ("title", "content")


@admin.register(CounselingSession)
class CounselingSessionAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "entry")
    list_filter = ("user",)


@admin.register(ChatTurn)
class ChatTurnAdmin(admin.ModelAdmin):
    list_display = ("id", "session", "sender", "created_at")
    list_filter = ("sender", "session__user")
    search_fields = ("message",)
