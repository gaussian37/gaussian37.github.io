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

<br>
    
### CRUD

+ All data can be managed by actions which Create/Read/Update/Delete records
    - C : Create 
    - R : Read 
    - U : Update
    - D : Delete

<br>

### REST API URL example

+ A Post model needs following function when it offers API services
    - Get new posting and registers it and respond (e.g. **/post/new/**)
    - respond posting list and search (e.g. request **GET on /post/ address**)
    - respond specific positing contents (e.g. **On 10th articles, Get on /post/10/**)
    - update specific posting contents and respond (e.g. **POST on /post/10/update**)
    - delete specific posting contents and respond (e.g. **POST on /post/10/delete**)

**DRF** is the framework offering `Class Based View` which helps create REST API.

<br>

### django rest framework setting

![flow](../assets/img/python/rest/JSON-Response-View/flow.png)

<br>

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

<br>

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
<br>

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

In the CLI, we can use `http` command.

For examples,

shell> http GET `request address` `GET parameter`==value `GET parameter`==값 <br>
shell> http --json POST `request address` `GET parameter`==value `GET parameter`==값 `POST parameter`==value `POST parameter`==value <br>
shell> http --form POST `request address` `GET parameter`==value `GET parameter`==값 `POST parameter`==value `POST parameter`==value <br>
shell> http PUT `request address` `GET parameter`==value `GET parameter`==값 `POST parameter`==value `POST parameter`==value <br>
shell> http DELETE `request address` `GET parameter`==value `GET parameter`==값 <br>

POST requests are classified two way.
+ In specifying `--form` : multipart/form-data request : identical to HTML format
+ In specifying `--json` or skipping it : application/json request : send the data with serialized JSON format

GET request

```
shell> http GET httpbin.org/get x==1 y==2

HTTP/1.1 200 OK
Access-Control-Allow-Credentials: true
Access-Control-Allow-Origin: *
Connection: keep-alive
Content-Length: 296
Content-Type: application/json
Date: Sat, 14 Oct 2017 18:37:49 GMT
Server: meinheld/0.6.1
Via: 1.1 vegur
X-Powered-By: Flask
X-Processed-Time: 0.000959157943726

{
    "args": {
        "x": "1",
        "y": "2"
    },
    "headers": {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "close",
        "Host": "httpbin.org",
        "User-Agent": "HTTPie/0.9.9"
    },
    "origin": "221.148.61.230",
    "url": "http://httpbin.org/get?x=1&y=2"
}
```
<br>

POST request

```
shell> http --form POST "httpbin.org/post" a=1 b=2 c=3

HTTP/1.1 200 OK
Access-Control-Allow-Credentials: true
Access-Control-Allow-Origin: *
Connection: keep-alive
Content-Length: 469
Content-Type: application/json
Date: Sat, 14 Oct 2017 18:43:14 GMT
Server: meinheld/0.6.1
Via: 1.1 vegur
X-Powered-By: Flask
X-Processed-Time: 0.00148487091064

{
    "args": {},
    "data": "",
    "files": {},
    "form": {
        "a": "1",
        "b": "2",
        "c": "3"
    },
    "headers": {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "close",
        "Content-Length": "11",
        "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        "Host": "httpbin.org",
        "User-Agent": "HTTPie/0.9.9"
    },
    "json": null,
    "origin": "221.148.61.230",
    "url": "http://httpbin.org/post"
}
```

<br>

PUT request

```
shell> http PUT httpbin.org/put hello=world

HTTP/1.1 200 OK
Access-Control-Allow-Credentials: true
Access-Control-Allow-Origin: *
Connection: keep-alive
Content-Length: 452
Content-Type: application/json
Date: Sat, 14 Oct 2017 18:37:05 GMT
Server: meinheld/0.6.1
Via: 1.1 vegur
X-Powered-By: Flask
X-Processed-Time: 0.00133204460144

{
    "args": {},
    "data": "{\"hello\": \"world\"}",
    "files": {},
    "form": {},
    "headers": {
        "Accept": "application/json, */*",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "close",
        "Content-Length": "18",
        "Content-Type": "application/json",
        "Host": "httpbin.org",
        "User-Agent": "HTTPie/0.9.9"
    },
    "json": {
        "hello": "world"
    },
    "origin": "221.148.61.230",
    "url": "http://httpbin.org/put"
}
```

<br>

DELETE request

```
shell> http DELETE "httpbin.org/delete"

HTTP/1.1 200 OK
Access-Control-Allow-Credentials: true
Access-Control-Allow-Origin: *
Connection: keep-alive
Content-Length: 351
Content-Type: application/json
Date: Sat, 14 Oct 2017 18:42:44 GMT
Server: meinheld/0.6.1
Via: 1.1 vegur
X-Powered-By: Flask
X-Processed-Time: 0.000683069229126

{
    "args": {},
    "data": "",
    "files": {},
    "form": {},
    "headers": {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "close",
        "Content-Length": "0",
        "Host": "httpbin.org",
        "User-Agent": "HTTPie/0.9.9"
    },
    "json": null,
    "origin": "221.148.61.230",
    "url": "http://httpbin.org/delete"
}
``` 