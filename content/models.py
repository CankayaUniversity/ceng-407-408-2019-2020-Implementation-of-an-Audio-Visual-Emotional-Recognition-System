import os
from django.db import models
from django.contrib.auth.models import User
from django.utils.safestring import mark_safe
from django.core.exceptions import ValidationError

LANGUAGES = (
    ('TR', 'Turkish'),
    ('EN', 'English'),
    ('FR', 'French')
)

TYPE = (
    ('Video', 'Video'),
    ('Image', 'Image'),
    ('Audio', 'Audio')
)

SEX = (
    ('Man', 'Man'),
    ('Women', 'Women')
)

EMOTIONS = (
    ('neutral', 'Neutral'),
    ('calm', 'Calm'),
    ('happy', 'Happy'),
    ('sad', 'Sad'),
    ('angry', 'Angry'),
    ('fearful', 'Fearful'),
    ('disgust', 'Disgust'),
    ('surprised', 'Surprised')
)


def validate_file_extension(value):
    ext = os.path.splitext(value.name)[1]
    valid_extensions = ['.mp4', '.wav']
    if not ext.lower() in valid_extensions:
        raise ValidationError(((mark_safe(
            '<p style="color:red;">Unsupported file!</p>'))))


def validate_file_size(value):
    filesize = value.size
    if filesize > 10485760:
        raise ValidationError(((mark_safe(
            "<p style='color:red;'>The maximum file size that can be uploaded is 10MB!</p>"))))
    else:
        return value


class Contents(models.Model):
    uploader = models.ForeignKey(User, on_delete=models.CASCADE, default=User)
    content_name = models.CharField(max_length=50)
    content_type = models.CharField(max_length=5, choices=TYPE, default=TYPE[0])
    content_language = models.CharField(max_length=7, choices=LANGUAGES, default=LANGUAGES[0])
    content_sex = models.CharField(max_length=5, choices=SEX, default=SEX[0])
    content_emotion = models.CharField(max_length=9, choices=EMOTIONS, default=EMOTIONS[0])
    commend = models.TextField()
    content_upload = models.FileField(null=False, upload_to="contents/",
                                      validators=[validate_file_extension, validate_file_size])
