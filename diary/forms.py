# diary/forms.py
from django import forms
from django.utils import timezone
from .models import DiaryEntry


class DiaryEntryForm(forms.ModelForm):
    # ✅ 위젯/포맷 커스터마이징용으로 필드 한 번 오버라이드
    entry_date = forms.DateField(
        label="일기 날짜",
        widget=forms.DateInput(
            attrs={
                "class": "form-control",
                "type": "date",
            },
            format="%Y-%m-%d",
        ),
        input_formats=["%Y-%m-%d"],
    )

    class Meta:
        model = DiaryEntry
        fields = ["entry_date", "title", "content", "mood_self_report"]
        widgets = {
            "title": forms.TextInput(attrs={"class": "form-control", "placeholder": "제목"}),
            "content": forms.Textarea(attrs={"class": "form-control", "rows": 12, "placeholder": "오늘의 일기를 적어주세요..."}),
            "mood_self_report": forms.NumberInput(attrs={"class": "form-control", "min": 1, "max": 10}),
        }
        labels = {
            "mood_self_report": "오늘의 기분(1~10)",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 새로 작성할 때만 기본값을 '오늘'로 세팅
        if not self.instance.pk:
            self.fields["entry_date"].initial = timezone.localdate()
