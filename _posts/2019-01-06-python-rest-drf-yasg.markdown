---
layout: post
title: django restframework API 문서 자동 생성
date: 2019-01-06 18:43:00
img: python/rest/rest.png
categories: [python-rest] 
tags: [python, django, REST, drf, restframework, yasg] # add tag
---

이번 글에서 알아볼 내용은 django restframework를 이용하여 API를 만들었을 때, 이것을 잘 설명할 수 있는 문서화를
자동으로 만드는 방법에 대하여 알아보겠습니다.

자동으로 문서화를 해줄 package는 `drf-yasg` 입니다.

django, django restframework에서 자동 문서화 도구가 여러가지 있었지만 `drf-yasg`가 많이 추천하는 분위기 였습니다.
그러면 간단하게 어떻게 사용하는지 알아보겠습니다.

### 패키지 설치

```python
pip install -U drf-yasg
pip install flex
```

<br>

+ def-yasg : API를 자동 문서화 해주는 패키지 입니다.
+ flex : JSON schema를 체크하는 검사 패키지 입니다.
    + JSON은 좀 더 쉽게 데이터를 교환하고 저장하기 위하여 만들어진 데이터 교환 표준입니다.
    + 이때 JSON 데이터를 전송받는 측에서는 전송받은 데이터가 적법한 형식의 데이터인지를 확인할 방법이 필요합니다.
    + 따라서 적법한 JSON 데이터의 형식을 기술한 문서를 JSON 스키마(schema)라고 합니다.

<br>

### 환경 설정

+ project/settings.py 의 INSTALLED_APPS에 추가

```python
INSTALLED_APPS = [
   ...
   'drf_yasg',
   ...
]
```

<br>

+ urls.py 에 아래 내용을 추가 합니다.

```python
from django.conf.urls import url
from drf_yasg.views import get_schema_view
from rest_framework.permissions import AllowAny
from drf_yasg import openapi
 
schema_url_v1_patterns = [
    url(r'^sample/v1', include('sample.urls', namespace='sample')),
]
 
schema_view_v1 = get_schema_view(
    openapi.Info(
        title="sample API",
        default_version='v1',
        description="This is the sample API for something",
        contact=openapi.Contact(email="service.sample@google.com"),
        license=openapi.License(name="sample company"),
    ),
    validators=['flex'],
    public=True,
    permission_classes=(AllowAny,),
    patterns=schema_url_v1_patterns,
)

urlpatterns = [
    ...
    url(r'^sample/v1/', include('sample.urls', namespace='sample_api')),
    ...
    
    # API document generation with  drf_yasg
    url(r'^v1/swagger(?P<format>\.json|\.yaml)/$', schema_view_v1.without_ui(cache_timeout=0), name='schema-json'),
    url(r'^v1/swagger/$', schema_view_v1.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    url(r'^v1/redoc/$', schema_view_v1.with_ui('redoc', cache_timeout=0), name='schema-redoc-v1'),
    ...
]
```

<br>

+ urls.py에 너무 코드가 길게 들어가는게 보기가 안좋습니다.

```python
schema_url_v1_patterns = [
    url(r'^sample/v1', include('sample.urls', namespace='sample')),
]
 
schema_view_v1 = get_schema_view(
    openapi.Info(
        title="sample API",
        default_version='v1',
        description="This is the sample API for something",
        contact=openapi.Contact(email="service.sample@google.com"),
        license=openapi.License(name="sample company"),
    ),
    validators=['flex'],
    public=True,
    permission_classes=(AllowAny,),
    patterns=schema_url_v1_patterns,
)
```

<br>

이 부분만 따로 떼어내어서 `import` 시키기를 권장 드립니다.

### API 문서 자동화에 사용 할 주석 작성

+ API 문서 자동화에 사용될 내용은 주석을 기반으로 작성 됩니다.
+ DRF의 `ViewSet` 함수의 `class 바로 아래` 또는 `각 method 바로 아래`에 작성하면 됩니다.
+ 각 함수 바로 앞에 주석을 사용하면 그 내용을 바탕으로 markdown 형식으로 입력됩니다.
    + 예를 들면 아래와 같습니다.

```python
class SampleViewSet(viewsets.ModelViewSet):
    '''
    Sample 관련 REST API 제공
    
    ---
    ... 내용 ...
    '''
    

    def list(self, request, *args, **kwargs):
        '''
        Sample의 list를 불러오는 API
        
        ---
        ... 내용 ...
```

<br>

위와 같이 `class 바로 아래` 또는 `method 바로 아래`에 작성하면 되고 주의할 점은 
+ 주석의 제목을 적고 : Sample 관련 REST API 제공
+ 한 줄 띄우고
+ 칸을 나누는 `---`를 입력하고 그 아래에
+ 내용을 적습니다. : ... 내용 ...

### API 문서 자동화 결과 보기

앞에서 url을 설정하였을 때 다음과 같이 url을 입력해 주었습니다.

```python
url(r'^v1/swagger(?P<format>\.json|\.yaml)/$', schema_view_v1.without_ui(cache_timeout=0), name='schema-json'),
url(r'^v1/swagger/$', schema_view_v1.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
url(r'^v1/redoc/$', schema_view_v1.with_ui('redoc', cache_timeout=0), name='schema-redoc-v1'),
```

<br>

즉, `redoc`과 `swagger` 라는 페이지가 각각 존재합니다.

+ `redoc`은 문서화에 좀 더 충실한 페이지이므로 문서화 목적으로 보려면 알맞습니다.
+ `swagger` 에서는 API를 호출해 볼 수 있습니다. 하지만 제 개인적으로는 `redoc`이 문서화에는 확실히 깔끔합니다.

![1](../assets/img/python/rest/yasg/sample.PNG)

위 이미지는 `redoc` 페이지를 접속하였을 때 예시입니다.

+ 참고 자료 : 
    + http://jay-ji.tistory.com/31
    + https://medium.com/towncompany-engineering/%EC%B9%9C%EC%A0%88%ED%95%98%EA%B2%8C-django-rest-framework-api-%EB%AC%B8%EC%84%9C-%EC%9E%90%EB%8F%99%ED%99%94%ED%95%98%EA%B8%B0-drf-yasg-c835269714fc