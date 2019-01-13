---
layout: post
title: django-debug-toolbar를 이용한 SQL 디버깅
date: 2019-01-14 18:43:00
img: python/django/django.png
categories: [python-django] 
tags: [python, django, django-debug-toolbar, sql, sql debugging] # add tag
---

이번 글에서는 `django-debug-toolbar`를 이용한 SQL 디버깅 하는 방법에 대하여 알아보도록 하겠습니다.

상세 내용을 위해서는 관련 공식 문서를 확인 하시면 됩니다.

+ https://django-debug-toolbar.readthedocs.io/en/latest/installation.html

이 때, 주의해야 할 사항은 웹페이지의 템플릿에 `<body>` 태그가 있어야만 `django-debug-toolbar`가 작동합니다.
    + `django-debug-toolbar`의 html/script 를 넣는 곳이 `<body>` 태그 이기 때문입니다.

<br><br>

## django-debug-toolbar 설치하기

+ pip를 이용한 패키지 설치하기

```python
pip install django-debug-toolbar
```

<br><br>

## django-debug-toolbar 환경 설정하기

+ project/settings.py 에 환경 구성하기
+ INSTALLED_APPS에 기본 APP으로 `django.contrib.staticfiles` 이 등록이 되어 있으나 혹시 빠져있으면 넣어주어야 합니다.
+ 설치한 `django-debug-toolbar`를 `debug_toolbar`로 구성해 줍니다.

```python
INSTALLED_APPS = [
    # ...
    'django.contrib.staticfiles',
    # ...
    'debug_toolbar',
]

STATIC_URL = '/static/'
```

<br>

+ project/urls.py 에서 debug 관련 url을 추가해 줍니다.
+ debug 관련 내용 이기 때문에 **반드시** `settings.DEBUG = True` 일 때만 사용하는 것을 추천드립니다.

```python
if settings.DEBUG:
    import debug_toolbar
    urlpatterns = + [
        path('__debug__/', include(debug_toolbar.urls)),

        # For django versions before 2.0:
        # url(r'^__debug__/', include(debug_toolbar.urls)),

    ]
```

<br>

+ `django-debug-toolbar`는 주로 `middleware`에서 동작합니다.
+ project/settings.py의 `MIDDLEWARE`에 debug toolbar를 추가합니다.

```python
MIDDLEWARE = [
    # ...
    'debug_toolbar.middleware.DebugToolbarMiddleware',
    # ...
]
```

<br>

+ 마지막으로 Internal IP를 추가해 줍니다.
+ 앞에서도 말씀드린 것과 같이 `django-debug-toolbar`를 이용하면 내부 상황을 다 볼 수 있습니다.
+ 따라서, 특정 IP에서만 사용할 수 있도록 권한을 관리해 주어야 합니다.
    + project/settings.py에 아래와 같이 localhost를 등록해 줍니다.
    
```python
# The Debug Toolbar is shown only if your IP is listed in the INTERNAL_IPS setting.
INTERNAL_IPS = ['127.0.0.1']
```

<br><br>

## django-debug-toolbar 실행 

+ 환경 구성을 완료하셨으면 웹페이지를 띄우면 바로 `django-debug-toolbar`가 실행됩니다.
+ 앞에서 설명드린 바와 같이 반드시 `<body> ... </body>`가 있어야 실행되게 됩니다.
+ 웹 페이지를 실행하면 아래와 같이 우측에 보이게 됩니다.
         
<img src="../assets/img/python/django/django-debug-toolbar/debug-toolbar.PNG" alt="Drawing" style="width: 500px;"/>

<br>

+ 특정 쿼리를 실행하게 되면 (웹페이지 조회) sql이 실행된 내용을 볼 수 있습니다.
+ 클릭하면 상세 내역도 볼 수 있습니다.

<img src="../assets/img/python/django/django-debug-toolbar/sql.PNG" alt="Drawing" style="width: 500px;"/>

도움이 되셨다면 광고 클릭 한번이 저에게 큰 도움이 됩니다. 감사합니다.