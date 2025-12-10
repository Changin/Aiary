# diary/tasks.py
from celery import shared_task
from .models import DiaryEntry, CounselingSession
from .services import bootstrap_counseling


@shared_task
def bootstrap_counseling_task(entry_id, session_id):
    entry = DiaryEntry.objects.get(pk=entry_id)
    session = CounselingSession.objects.get(pk=session_id)
    bootstrap_counseling(entry, session)
