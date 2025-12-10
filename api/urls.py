# api/urls.py

from django.urls import path
from . import views

app_name = "api"

urlpatterns = [
    path("chat/send/", views.chat_send, name="chat_send"),
    path("ocr/recognize/", views.ocr_recognize, name="ocr_ocr_recognize")
]
