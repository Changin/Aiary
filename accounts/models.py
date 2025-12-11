from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

# Create your models here.
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    # 기본: 무료 일반 유저 → 턴 제한 적용
    turn_limit_enabled = models.BooleanField(default=True)
    # 향후 확장: 유료 플랜은 세션당 턴 수 조정 가능
    max_turns_per_session = models.PositiveIntegerField(null=True, blank=True)
    # 향후 확장: 토큰 구입한 경우 남은 토큰 수
    token_balance = models.PositiveIntegerField(null=True, blank=True)

    def get_max_turns(self, default=5):
        if self.turn_limit_enabled is False:
            return None  # 제한 없음
        if self.max_turns_per_session:
            return self.max_turns_per_session
        return default


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)
