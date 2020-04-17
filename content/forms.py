from django import forms
from .models import Contents


class ContentsForm(forms.ModelForm):
    class Meta:
        model = Contents
        fields = ['uploader', 'content_name', 'content_type', 'content_language', 'content_sex', 'content_emotion',
                  'commend', 'content_upload']
