---
layout: post
title: JSON 응답뷰 만들기
date: 2018-09-16 18:43:00
img: python/rest/rest.png
categories: [python-rest] 
tags: [python, django, rest] # add tag
---

### API 서버는 무엇일까요?

+ 앱/웹 서비스를 만드는 **개발자**들이 이용하는 `데이터 위주`의 서비스 
+ API 서버는 시간이 지나도 호환성을 유지해야 합니다.
    - 따라서 API 버전의 개념을 두어야 합니다.
        - ex) /api/v1/posts, /api/v2/posts/

+ django`rest`framework(이하 `DRF`)는 REST API를 만드는 데 도움을 줍니다.
    - URI는 `https://{serviceRoot}/{collection}/{id}' 형식이어야 합니다.
    - DRF는 GET, PUT, DELETE, POST, HEAD, PATCH, OPTIONS 기능을 지원해야 합니다.
    - API 버전은 `Major.minor` 형태로 관리하고 URI에 버전 정보를 포함시켜야 합니다.
+ DRF는 장고의 Form/CBV을 컨셉을 그대로 가져왔습니다. 따라서 장고 Form/CBV를 잘 이해한다면 DRF에 대해서도 보다 깊은 이해가 가능합니다. 

<br>
    
### CRUD

+ 모든 데이터는 기본적으로  Create/Read/Update/Delete 로 관리가 되어야 합니다.
    - C : Create (레코드 생성)
    - R : Read (레코드 목록 조회, 특정 레코드 상세 조회)
    - U : Update (특정 레코드 수정)
    - D : Delete (특정 레코드 삭제)

<br>

### REST API URL 예제

+ REST API 식의 URL로 설계를 해보면 다음과 같이 해볼 수 있습니다.
    + /post/주소
        + GET 방식 요청 : 목록 응답
        + POST 방식 요청 : 새 글 생성하고, 확인 응답
    + /post/1/주소
        + GET 방식 요청 : 1번글 응답
        + PUT 방식 요청 : 1번글 갱신하고 확인 응답
        + DELETE 방식 요청 : 1번글 삭제하고 확인 응답

+ django`rest`framework는 REST API 구현을 도와주는 Class Based View를 제공해주는 프레임워크입니다. 

<br>

### django rest framework setting

![flow](../assets/img/python/rest/JSON-Response-View/flow.png)

<br>

+ 먼저 프로젝트를 하나 만들어 보겠습니다. 커맨드 라인에 다음과 같이 입력하면 프로젝트가 생성됩니다.

```python
django-admin startproject djangoRestApi
cd/djangoRestApi
```

<br>

+ 프로젝트에서 사용할 앱을 만들어 보겠습니다. 앱 이름은 sample로 지정해보면 아래와 같습니다.
```python
python manage.py startapp sample
```

<br>

djangoRestApi/settings.py, 에서 `INSTALLED_APPS`에 `rest_framework`, `sample`.를 추가해 보겠습니다.

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

+ 앱에서 모델을 추가해 보겠습니다.

```python
# sample/models.py
from django.db import models

class Post(models.Model):
    message = models.TextField()
```

<br>

+ 같은 디렉토리에서 다음과 같이 `serializers.py`를 만들고 안에 코드를 추가해 보겠습니다.

```python
# myapp/serializers.py
from rest_framework import serializers
from .models import Post

# ModelSerializer (instead of ModelForm)
class PostSerializer(serializers.ModelSerializer):
    class Meta:
        model = Post
        fields = '__all__'
```

<br>

+ views.py에 Serializers.py에서 작성한 내용을 불러오겠습니다.

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

+ sample/urls.py에서 views.py에 선언된 값들을 url로 연결해줍니다.

```python
# sample/urls.py
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'posts', views.PostViewSet)

urlpatterns = [
    url(r'', include(router.urls)),
]
```
<br>

### API 호출

+API 뷰 호출은 다양한 클라이언트 프로그램에 의하여 호출될 수 있습니다.
    + JavaScript from web frontend
    + Android/iOS app code
    + Library (ex.requests)
    + CLI Program : cURL, HTTPie

### HTTPie 란?

+ 다음 명령어를 통해 간단히 설치할 수 있습니다.

```python
pip install --upgrade httpie
```

<br>

+ 설치를 하면 CLI에서 `http`라는 명령어를 사용할 수 있습니다.
+ 예를 들면 다음과 같습니다.

```python
shell> http GET `request address` `GET parameter`==value `GET parameter`==값 <br>
shell> http --json POST `request address` `GET parameter`==value `GET parameter`==값 `POST parameter`==value `POST parameter`==value <br>
shell> http --form POST `request address` `GET parameter`==value `GET parameter`==값 `POST parameter`==value `POST parameter`==value <br>
shell> http PUT `request address` `GET parameter`==value `GET parameter`==값 `POST parameter`==value `POST parameter`==value <br>
shell> http DELETE `request address` `GET parameter`==value `GET parameter`==값 
```

<br>

+ 참고로 GET 인자에는 `==`을 사용하고 POSㅆ 인자에는 `=`를 사용하였습니다.

+ POST requests는 다음과 같이 두 가지 방식으로 나뉘어 집니다.
    + In specifying `--form` : multipart/form-data 요청 : HTML Form과 동일합니다.
    + In specifying `--json` or 생략 시 : application/json 요청 : 요청 데이터를 JSON포맷으로 직렬화해서 전달합니다.

<br>

+ GET request 예시

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

+ POST request 예시

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

+ PUT request 예시

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

+ DELETE request 예시

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

