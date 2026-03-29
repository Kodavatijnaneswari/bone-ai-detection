from django import forms
from django.core.exceptions import ValidationError
import re
from .models import modeldata

class modeldataForm(forms.ModelForm):
    class Meta:
        model = modeldata
        fields = ['name', 'username', 'password', 'mobile', 'email', 'address']
        widgets = {
            'password': forms.PasswordInput(),
            'address': forms.Textarea(attrs={'rows': 3, 'cols': 40}),
        }

    def clean_name(self):
        name = self.cleaned_data.get('name') or ""
        if not re.match(r'^[a-zA-Z\s]+$', name):
            raise ValidationError('Name should contain only letters and spaces.')
        return name

    def clean_username(self):
        username = self.cleaned_data.get('username') or ""
        if not re.match(r'^[a-zA-Z0-9_.-]+$', username):
            raise ValidationError('Username can contain only letters, numbers, dots, underscores, and hyphens.')
        return username

    def clean_password(self):
        password = self.cleaned_data.get('password') or ""
        if not re.match(r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@$!%*?&]{8,}$', password):
            raise ValidationError(
                'Password must be at least 8 characters long and include at least one letter and one number.'
            )
        return password

    def clean_mobile(self):
        mobile = self.cleaned_data.get('mobile') or ""
        if not re.match(r'^\d{10}$', mobile):
            raise ValidationError('Mobile number must be exactly 10 digits.')
        return mobile

    def clean_email(self):
        email = self.cleaned_data.get('email') or ""
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            raise ValidationError('Enter a valid email address.')
        return email

    def save(self, commit=True):
        instance = super().save(commit=False)
        # Ensure status is always set to 'waiting'
        instance.status = 'waiting'
        if commit:
            instance.save()
        return instance
