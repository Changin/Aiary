from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from django.urls import reverse_lazy
from django.views.generic import CreateView

# Create your views here.


# 회원가입 뷰
class SignupView(CreateView):
    form_class = UserCreationForm
    template_name = "accounts/signup.html"
    success_url = reverse_lazy("web:home")

    def form_valid(self, form):
        # 유저 생성
        response = super().form_valid(form)
        # 회원가입 직후 자동 로그인
        user = self.object
        login(self.request, user)
        return response
