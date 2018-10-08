---
layout: post
title: Create JSON Response view
date: 2018-09-16 18:43:00
img: python/rest/rest.png
categories: [python-rest] 
tags: [python, django, rest] # add tag
---

### What is API server?

+ Data oriented services for **developer** who make web/app services.
+ API server must maintain compatability over time.
    - Accordingly, API sever has version.
        - ex) /api/v1/posts, /api/v2/posts/

+ django`rest`framework(in short, DRF) helps to make REST API concept below.
    - URI format is `https://{serviceRoot}/{collection}/{id}'
    - DRF support GET, PUT, DELETE, POST, HEAD, PATCH, OPTIONS.
    - API vesion is Major.minor and is included in URI
    
### CRUD

+ All data can be managed by actions which Create/Read/Update/Delete records
    - C : Create 
    - R : Read 
    - U : Update
    - D : Delete

### REST API URL example

+ A Post model needs following function when it offers API services
    - Get new posting and registers it and respond (ex. /post/new/)
    - respond posting list and search (ex. request GET on /post/ address)
    - respond specific positing contents (ex. On 10th articles, Get on /post/10/)
    - update specific posting contents and respond (ex. POST on /post/10/update)
    - delete specific posting contents and respond (ex. POST on /post/10/delete)

**DRF** is the framework offering `Class Based View` which helps create REST API.

### django rest framework setting

First, create project in command line. Any project name is ok.

```python
django-admin startproject djangoRestApi
cd/djangoRestApi
```

<br>

create app in command line. app name is sample
```python
python manage.py startapp sample
```

<br>

In djangoRestApi/settings.py, add `rest_framework`, `sample`.

```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
	'rest_framework',
	'sample',
]
```

```python
# myapp/models.py
from django.db import models

class Post(models.Model):
    message = models.TextField()
```

<br>

In the sample directory, set the files as below.

```python
# myapp/serializers.py
from rest_framework import serializers
from .models import Post
```

<br>

```python
# ModelSerializer (instead of ModelForm)
class PostSerializer(serializers.ModelSerializer):
    class Meta:
        model = Post
        fields = '__all__'
```

<br>

```python
# myapp/views.py
from django.shortcuts import render
from rest_framework import viewsets
from .models import Post
from .serializers import PostSerializer

class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.all()
    serializer_class = PostSerializer
```

<br>

```python
# myapp/urls.py
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'posts', views.PostViewSet)

urlpatterns = [
    url(r'', include(router.urls)),
]
```

### API Call

API view can be called by various client program.

+ JavaScript from web frontend
+ Android/iOS app code
+ Library (ex.requests)
+ CLI Program : cURL, HTTPie

### What is the HTTPie ?

You can install easily

```python
pip install --upgrade httpie
```





