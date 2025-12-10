# api/urls.py

from django.urls import path
from . import views

app_name = "api"

urlpatterns = [
    path("chat/send/", views.chat_send, name="chat_send"),
    path("chat/history/", views.chat_history, name="chat_history"),
    path("ocr/start/", views.ocr_start, name="ocr_start"),
    path("ocr/result/", views.ocr_result, name="ocr_result"),
]
