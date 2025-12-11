# diary/forms.py
from django import forms
from django.utils import timezone
from .models import DiaryEntry


class DiaryEntryForm(forms.ModelForm):
    # 위젯/포맷 커스터마이징용으로 필드 한 번 오버라이드
    entry_date = forms.DateField(
        label="일기 날짜",
        widget=forms.DateInput(
            attrs={
                "id": "date",
                "class": "form-control",
                "type": "date",
                "style": "border-radius: 16px; padding: 10px 15px; font-size: 1rem; border: 1px solid #a7aaae;",
            },
            format="%Y-%m-%d",
        ),
        input_formats=["%Y-%m-%d"],
    )

    class Meta:
        model = DiaryEntry
        fields = ["entry_date", "title", "content", "mood_self_report"]
        widgets = {
            "title": forms.TextInput(attrs={"id": "title", "class": "form-control", "placeholder": "제목을 입력하세요.", "style": "color: #1A1A1A;border-radius: 16px;padding: 10px 15px;font-size: 1rem;border: 1px solid #a7aaae;"}),
            "content": forms.Textarea(attrs={"id": "content", "class": "form-control", "rows": 12, "style": "border-radius: 16px;color: #1a1a1a;padding: 15px;font-size: 1rem;border: 1px solid #a7aaae;min-height: 250px;"}),
            "mood_self_report": forms.NumberInput(attrs={"id": "mood", "class": "form-control", "min": 1, "max": 10, "step": 1, "style": "border-radius: 16px;color: #1a1a1a;padding: 10px 15px;font-size: 1rem;border: 1px solid #A7AAAE;"}),
            "contents_predicted": forms.HiddenInput(),  # 숨김 필드 (OCR모델이 예측한 값 POST로 넘어감)
        }
        labels = {
            "mood_self_report": "오늘의 기분(1~10)",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 새로 작성할 때만 기본값을 '오늘'로 세팅
        if not self.instance.pk:
            self.fields["entry_date"].initial = timezone.localdate()
