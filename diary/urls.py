# diary/urls.py

from django.urls import path
from . import views
from django.http import HttpResponse

app_name = "diary"


def placeholder(request):
    return HttpResponse("diary placeholder (추후 구현 예정)")


urlpatterns = [
    path("", views.EntryListView.as_view(), name="list"),
    path("new/", views.EntryCreateView.as_view(), name="create"),
    path("<int:pk>/", views.EntryDetailView.as_view(), name="detail"),  # 상세페이지
    path("<int:pk>/delete/", views.EntryDeleteView.as_view(), name="delete"),   # 일기 삭제
    path("summary/", views.EntrySummaryView.as_view(), name="summary"),     # 요약 페이지

    # 임시 placeholder
    # path("", placeholder, name="list"),
    # path("new/", placeholder, name="create"),
]
