# Aiary/celery.py
import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Aiary.settings")

app = Celery("Aiary")

# Redis 브로커 + 결과백엔드
app.conf.broker_url = "redis://localhost:6379/0"
app.conf.result_backend = "redis://localhost:6379/1"

# Django settings의 CELERY_ 로 시작하는 값 자동 로드 (있으면)
app.config_from_object("django.conf:settings", namespace="CELERY")

# 각 앱의 tasks.py 자동 검색
app.autodiscover_tasks()
