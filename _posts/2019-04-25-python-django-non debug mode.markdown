---
layout: post
title: Debug=False 일 때, 발생할 수 있는 오류
date: 2019-04-25 00:00:00
img: python/django/django.png
categories: [python-django] 
tags: [python, django, Debug] # add tag
---

+ 서버를 public으로 사용할 때 project/settings.py에서 반드시 `Debug=False`로 바꾸어 주어야 합니다.
    + Debug 모드일 때에는 프로젝트의 url 구조를 알 수 있기 때문에 공개되어서는 안됩니다.
+ 개발 시에는 문제가 되지 않았다가 배포시에 문제가 발생할 수 있는 몇가지가 있어서 정리를 해두려고 합니다.

### media, static 파일 url 경로 추가

+ Debug=True 모드일 때에는 url에 직접적으로 등록되지 않은 링크는 Error log가 뜨거나 일부 접속되는 항목이 있습니다.
+ 여기서 일부 접속되는 항목이 대표적으로 media, static 파일과 같은 경우 입니다.
    + 예를 들어 `http://localhost:8000/media/profile/2019/04/23/sample.png` 라는 링크가 있다고 하겠습니다.
    + 이 때, project/urls.py에 `http://localhost:8000/media/` 관련 링크가 없더라도 실제로 파일이 디스크에 저장되어 있으므로 ImageField를 통하여 링크가 연결이 됩니다.
    + 하지만 `Debug=False`에서는 urls.py에 등록되어 있지 않으면 `Not Found`에러가 발생합니다.

<br>

+ 해결 방법은 다음과 같습니다.
+ project/urls.py에 다음 라이브러리를 추가합니다.

```python
from django.views.static import serve
```

<br>

+ project/urls.py의 `urlpatterns`에 다음 `url`을 추가합니다.

```python
url(r'^media/(?P<path>.*)$', serve,{'document_root': settings.MEDIA_ROOT}),
url(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}),
```

<br>

+ 위 라이브러리와 코드를 추가하면 정상적으로 media파일과 static파일에 관한 링크를 Debug 모드가 아닐 때에도 불러올 수 있습니다.
