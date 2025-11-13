# api/urls.py

from django.urls import path
from . import views

app_name = "api"

urlpatterns = [
    path("chat/send/", views.chat_send, name="chat_send"),
    # path("analysis/stream/", views.analysis_stream, name="analysis_stream"),  # 6단계에서 구현
    # path("chat/stream/", views.chat_stream, name="chat_stream"),
]
