from django.db import models
from django.contrib.auth.models import User

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


class Contents(models.Model):
    uploader = models.ForeignKey(User, on_delete=models.CASCADE, default=User)
    content_name = models.CharField(max_length=50)
    content_type = models.CharField(max_length=5, choices=TYPE, default=TYPE[0])
    content_language = models.CharField(max_length=7, choices=LANGUAGES, default=LANGUAGES[0])
    content_sex = models.CharField(max_length=5, choices=SEX, default=SEX[0])
    content_emotion = models.CharField(max_length=9, choices=EMOTIONS, default=EMOTIONS[0])
    commend = models.TextField()
    content_upload = models.FileField(null=False, upload_to="contents/")
