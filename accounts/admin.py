from django.contrib import admin
from django.contrib.auth.models import User
from .models import Profile


class ProfileInline(admin.StackedInline):
    model = Profile
    can_delete = False


class UserAdmin(admin.ModelAdmin):
    inlines = [ProfileInline]
    list_display = ("username", "email", "is_staff")
    search_fields = ("username", "email")


admin.site.unregister(User)
admin.site.register(User, UserAdmin)
