# Implementation of an Audio-Visual Emotional Recognition System

### Compilation / Installation Guide

```console
$ git clone <project web URL> <project name>
$ cd <project name>
$ pip3 install virtualenv
$ virtualenv venv
$ source venv/bin/activate
$ pip3 install -r requirements.txt
$ sudo apt install ffmpeg
$ python manage.py makemigrations
$ python manage.py migrate
$ python manage.py runserver
```
You want to see some properties to this system, you can create a superuser. You can follow the below comments:
```console
$ python manage.py createsuperuser
```

#### Note:
- This system work with PostgreSql database. 
Please install PostgreSql and configure with django settings.py. 
You may follow this tutorial for installations: [Tutorial](https://www.digitalocean.com/community/tutorials/how-to-use-postgresql-with-your-django-application-on-ubuntu-14-04)
- Please contact for the weights of the model.
