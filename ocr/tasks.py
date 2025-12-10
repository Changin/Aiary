# ocr/tasks.py - ocr용 celery 태스크 정의

import os
from celery import shared_task
from django.conf import settings
from .engine import initialize_set, process_full_ocr

_ENGINE_INITIALIZED = False


# OCR 엔진 활성화, 한번만 실행
def _ensure_initialized():
    global _ENGINE_INITIALIZED
    if not _ENGINE_INITIALIZED:
        initialize_set(settings.OCR_MODEL_DIR)
        _ENGINE_INITIALIZED = True


@shared_task
def run_ocr_task(image_path: str) -> str:
    """
    image_path: 실제 파일 경로 (예: MEDIA_ROOT/diary_ocr/xxx.jpg)
    """
    _ensure_initialized()
    try:
        text = process_full_ocr(image_path)
        return text or ""
    finally:
        # 처리 후 이미지 파일 삭제
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception:
            pass
