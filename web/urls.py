# web/urls.py

from django.urls import path
from django.views.generic import TemplateView
from . import views

app_name = "web"
urlpatterns = [
    # path("", views.HomeView.as_view(), name="home")
    path("", TemplateView.as_view(template_name="web/home.html"), name="home")  # 임시 html 뷰
]
