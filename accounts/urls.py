# accounts/urls.py

from django.urls import path
from .views import SignupView
from django.contrib.auth import views as auth_views
from django.http import HttpResponse

app_name = "accounts"


# def placeholder(request):
#   return HttpResponse("accounts placeholder (추후 구현 예정)")


urlpatterns = [
    path("login/", auth_views.LoginView.as_view(template_name="accounts/login.html"), name="login"),
    path("logout/", auth_views.LogoutView.as_view(next_page="web:home"), name="logout"),
    path("signup/", SignupView.as_view(), name="signup"),

    # 오류 안나게 placeholder 임시 구현
    # path("login/", placeholder, name="login"),
    # path("logout/", placeholder, name="logout"),
    # path("signup/", placeholder, name="signup"),
]
