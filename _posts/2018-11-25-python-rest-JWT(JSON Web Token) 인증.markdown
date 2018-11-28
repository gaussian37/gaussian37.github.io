---
layout: post
title: DRF에서 JWT(JSON Web Token) 사용하는 방법
date: 2018-11-25 18:43:00
img: python/rest/rest.png
categories: [python-rest] 
tags: [python, django, REST, JWT, JSON Web Token] # add tag
---

+ DRF(Django Rest Framework)에서 기본 적으로 지원하는 **Token**은 단순한 랜덤 문자열 입니다.
    - 각 User와 1:1 매칭을 이용합니다.
    - 유효 기간이 없습니다.
    
```python
import binascii
import os

binascii.hexlify(b'12345').decode()

>> '3132333435'
```

<br>

즉, DRF 기본 Token은 랜덤 문자열 이므로 의미 있는 데이터를 가지고 있지 않습니다. <br>

반면에, JWT는 **토큰 자체가 데이터**를 가지고 있습니다.

JWT는

+ 또 다른 형태의 Token 입니다.
+ 데이터베이스를 조회하지 않아도 **로직만으로도 인증이 가능**합니다.
+ 토큰 포맷 : **헤더.내용.서명**
+ 서버 측에서 토큰 발급 시에 비밀키로 서명을 하고, **발급시간** claim으로서 exp를 사용합니다.
    + claim은 Key/value 형식의 정보입니다.
    + 토큰을 요청한 유저 측에서는 `유저네임`과 `패스워드`를 제공을 해야하고, 유저네임과 패스워드가 맞으면 Token을 발급해 주는데, <br>
      이 때, 서버에서 임의로 정한 비밀키로 서명을 하고 `발급시간`을 `exp` 라는 필드에 담아서 발급을 해줍니다.
      
+ 서명을 하는 것일 뿐, 암호화는 아닙니다. 누구라도 열어볼 수 있으므로 보안에 중요한 데이터를 넣으면 안됩니다. 필요한 최소한의 정보만 담는 것이 권장됩니다.
    + 인증 요청 시 마다 토큰을 보내야 하므로 필요 없는 정보가 포함되면 매 번 비용이 많아지므로 최소한의 정보만 담는게 좋습니다.
+ djangorestframework-jwt에서는 Payload 항목에 디폴트로 user_id, user_name, e-mail 이름의 claim을 사용합니다.
    + id / name / mail 정보가 Token 자체에 있으므로 DB를 조회하지 않아도 누구인지 인증이 가능합니다.
    + `Payload = id + name + e-mail + exp` 이고 토큰 포맷의 **헤더.내용.서명** 중에 **내용**에 해당합니다.
    
+ 갱신(Refresh) 매커니즘을 지원합니다.
    + Token 유효기간 내에 갱신하거나, 유저네임/패스워드를 통해 재 인증 받아야 합니다.
    
### - Token은 반드시 안전한 장소에 보관해야 합니다.

+ 일반 Token / JWT 토큰 여부에 상관 없이 Token은 반드시 안전하게 보관 되어야 합니다. (인증 관련 요소이므로)
+ 스마트폰 앱은, 설치된 앱 별로 안전한 저장 공간이 제공되므로 토큰 방식을 쓰는 것이 좋지만 웹브라우저를 사용할 때는 저장 공간이 없습니다.
    + **Token은 앱 환경**에서만 권장 됩니다.
    + **웹 클라이언트 환경**에서는 보안적인 측면에서 **세션 인증**이 나은 선택일 수도 있습니다.  
    
   
### - Token 예시

8df73dafbde4c669dc37a9ea7620434515b2cc43

### - JWT 예시

ex) J0eXAi...IUzI1NiJ9`.`eyJ1c2VyX2lkIjo...aWwiOiIifQ`.`Zf_o3S7Q7-cmUz...LcF-2VdokJQ (너무 길어서 ... 로 줄였습니다.)

+ JWT는 **.** 을 기준으로 3영역으로 나뉘게 됩니다.
+ 헤더(Header)를 base64 인코딩하여, eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9
+ 내용(Payload)를 base64 인코딩하여, eyJ1c2VyX2lkIjoxLCJ1c2VybmFtZSI6ImFza2RqYW5nbyIsImV4cCI6MTUxNTcyMTIxMSwiZW1haWwiOiIifQ
+ 서명(Signature) : Header/Payload를 조합하고 비밀키로 서명한 후 base64 인코딩하여, Zf_o3S7Q7-cmUzLWlGEQE5s6XoMguf8SLcF-2VdokJQ

### - DRF에 JWT 세팅하기

+ 먼저 패키지를 설치 합니다.
    + pip install djangorestframework-jwt

+ 프로젝트/urls.py 에서 패키지 및 url을 등록해줍니다.
    + obtain_jwt_token : JWT 토큰 획득
    + refresh_jwt_token : JWT 토큰 갱신
    + verify_jwt_token : JWT 토큰 확인

```python
project/urls.py

from rest_framework_jwt.views import obtain_jwt_token, refresh_jwt_token, verify_jwt_token

urlpatterns = [
    ... ,
    url(r'^api-jwt-auth/$', obtain_jwt_token),          # JWT 토큰 획득
    url(r'^api-jwt-auth/refresh/$', refresh_jwt_token), # JWT 토큰 갱신
    url(r'^api-jwt-auth/verify/$', verify_jwt_token),   # JWT 토큰 확인
]
```

<br>

+ 프로젝트/settings.py 에서 REST_FRAMEWORK 의 세팅값을 수정해 줍니다.

```python
REST_FRAMEWORK = {
    ...
    'DEFAULT_AUTHENTICATION_CLASSES' : [
        ...
        'rest_framework_jwt.authentication.JSONWebTokenAuthentication',
    ]
}
```

<br>

+ 프로젝트/settings.py 에서 JWT_AUTH 에서 JWT 갱신 허용을 `True`로 설정해 줍니다.

```python
JWT_AUTH = { 
    'JWT_ALLOW_REFRESH': True, 
}
```

<br>

### - HTTPie를 통한 JWT 발급

다양한 언어 또는 방법을 통해서 특정 URL에 `POST`를 하는 방법이 있습니다.
그 중, `HTTPie`를 이용하여 `POST`를 해보겠습니다.

```python
http POST http://localhost:8000/api-jwt-auth/ username="유저명" password="암호"
{
    "token": "인증에 성공할 경우, 토큰응답이 옵니다."
}    
```

<br>

+ 이 때 사용할 유저명과 암호는 예를 들어서 createsuperuser로 만들었던 유저명과 암호를 사용하시면 실습해 볼 수 있습니다.
+ 인증에 실패할 경우 `400 Bad Request` 응답을 받습니다.
+ 발급 받은 JWT Token을 https://jwt.io/ 에 입력하면 어떤 의미를 가지고 있는지 알 수 있습니다.
    + jwt.io에서 Signature를 인증할 때 Seceret Code는 프로젝트/settings.py에 있는 `SECRET_KEY`를 입력하면 됩니다.
+ 정상적으로 POST가 되면 다음과 같은 Token이 발급 됩니다.
    + ex) eyJ0eXAiO3JKV2QeL...
    + 발급 받은 Token의 형태를 보면 앞에서 언급한 바와 같이 .을 기준으로 3부분으로 구분됩니다.    

### - 발급받은 JWT Token 확인

verify 를 통하여 JWT Token으로 인증이 잘 되는지 확인할 수 있습니다.

```python
http POST http://localhost:8000/api-jwt-auth/verify/ token="eyJ0eXAiO3JKV2QeLCJhdGci..."
```

<br>

+ 이 때 인증이 잘 되면 입력한 Token이 그대로 반환 됩니다. ex) eyJ0eXAiO3JKV2...
+ 만약 exp가 만료가 되었다면 다음과 같이 반환 됩니다.

```python
{
    "detail": "Signature has expired."
}

```

<br>

### - 발급받은 JWT Token으로 포스팅 목록 API 요청

Authorization이 필요한 Application을 접근할 때 JWT를 이용하여 ListView를 하려면 아래와 같이 할 수 있습니다.
아래에서 app/post/는 예시 입니다. "Authorization: JWT 토큰"에서 토큰 부분에 실제 토큰을 입력하면 됩니다.
ex) "Authorization: JWT eyJ0eXAiO3JKV2..."

```python
http http://localhost:8000/app/post/ "Authorization: JWT 토큰"
```

<br>

정리를 하면, 
+ DRF Token에서는 인증헤더 시작문자열로 "Token"을 사용하였지만, JWT 에서는 "JWT"를 사용합니다.
+ 인증이 성공할 경우 해당 API의 응답을 받습니다.
+ **이제 매 API 요청마다, 필히 JWT 토큰을 인증헤더에 담아 전송해야 합니다.**

### - JWT Token 유효기간이 지났을 경우

앞에서 설명드린 바와 같이 JWT Token의 유효기간이 지났을 경우에는 다음과 같이 반환받습니다.

```python
http http://localhost:8000/app/post/ "Authorization: JWT 토큰"

HTTP/1.0 401 Unauthorized
{ 
    "detail": "Signature has expired."
}
```

<br>

+ 클라이언트(앱) 에서는 토큰 유효기간 내에 토큰을 갱신 받아야 합니다. 유효 기간이 지났을 경우, 위와 같이 "401 Unauthorized" 응답을 받게 됩니다. 
+ **유효 기간 내**에는 **Token 만**으로 갱신을 받을수 있습니다.
+ **유효 기간이 지나면** `username/password`를 통해 인증 받아야 합니다.
+ JWT 토큰 유효기간은 `settings.JWT_AUTH`의 `JWT_EXPIRATION_DELTA` 값을 참조 하며 기본 값은 5분 입니다.

### - JWT Token 갱신받기

반드시 Token 유효 기간 내에 갱신이 이루어 져야 합니다.

```python
http POST http://localhost:8000/api-jwt-auth/refresh/ token="토큰"
{ 
    "token": "갱신받은 JWT 토큰" 
}
```

<br>
<br>

+ settings.JWT_AUTH의 `JWT_ALLOW_REFRESH` 설정은 디폴트가 `False` 입니다. `True` 설정을 해야 갱신을 진행할 수 있습니다.


### - djangorestframework-jwt의 주요 settings

```python
JWT_AUTH = { 
    'JWT_SECRET_KEY': settings.SECRET_KEY, 
    'JWT_ALGORITHM': 'HS256', 
    'JWT_EXPIRATION_DELTA': datetime.timedelta(seconds=300), 
    'JWT_ALLOW_REFRESH': False, 
    'JWT_REFRESH_EXPIRATION_DELTA': datetime.timedelta(days=7), 
}
```

<br>

기본적인 Setting 값은 위와 같습니다. 여기서 바꿔줘야 하는 것은 `JWT_EXPIRATION_DELTA`, `JWT_ALLOW_REFRESH`, `JWT_REFRESH_EXPIRATION_DELTA` 항목 입니다.
+ `JWT_EXPIRATION_DELTA` : Token 만료시간으로 기본값은 5분 입니다. 5분은 너무 짧으니 늘려 줍시다.
+ `JWT_ALLOW_REFRESH` : Token Refresh가 가능하도록 해줘야 합니다. 따라서 True로 바꾸어 줍니다.
+ `JWT_REFRESH_EXPIRATION_DELTA` : Refresh 가능 시간 입니다. 상식적으로 만료 되기전에 Refresh를 하는게 맞으므로 위의 `JWT_EXPIRATION_DELTA` 시간보다 짧게 해주면 됩니다.

<br><br>

## 실제 SNS랑 연동해서 사용해 보는 방법에 대하여 본격적으로 확인해 봅시다.

카카오를 예를 들어 설명해 보겠습니다. 카카오로부터 `Access Token`을 획득하고, 이를 장고 서버를 통해 `JWT 토큰`을 획득해야 합니다.

+ 앱 ~ **카카오톡** 서버
    + 안드로이드 앱에서 "카카오톡 로그인" 버튼을 클릭하면, 카카오톡 서버와 인증을 수행합니다.
    + 카카오톡 서버와의 인증에 성공하면, 카카오톡으로부터 Access Token 을 획득 합니다.
        + Access Token은 안드로이드/ios에서 `Native API`를 통해 얻습니다. 이 또한 만료 시간은 있지만 만료 시간이 깁니다.

+ 앱 ~ **장고** 서버
    + 획득된 Access Token을 장고 서버 인증 Endpoint (/accounts/rest-auth/kakao/)를 통해, JWT 토큰 획득
        + /accounts/rest-auth/kakao/ 는 django 서버에서 따로 설정해 주어야 하는 URL 입니다.
        + kakao 말고 따른 SNS 를 추가 하고 싶으면 /accounts/rest-auth/facebook/ 과 같이 설정할 수 있도록 유연하게 설계하면 됩니다.
    + 획득한 JWT 토큰이 만료되기 전에, `갱신` 합니다.
    + 획득한 JWT 토큰이 만료되었다면, Access Token을 서버로 전송하여 `JWT 토큰 재획득`
    
### 1) 카카오 개발자 홈페이지에서 필요한 정보를 가져옵니다.

https://developers.kakao.com 를 접속하여 로그인한 후에 Application을 하나 만들어 보겠습니다.

![1](../assets/img/python/rest/JWT/kakao1.PNG)

여기서 `네이티브 앱 키`는 스마트폰 앱에서 사용할 것이고 REST API 키는 장고 서버에서 사용할 것입니다.
`네이티브 앱 키`는 정확히 말하면 안드로이드/아이폰 앱 등에서 `kakao_app_key` 값으로 활용됩니다.

![2](../assets/img/python/rest/JWT/kakao2.PNG)

사용자 관리탭에서 사용자 관리 사용 옵션을 `ON`으로 켜주고, 수집 목적등을 기입한 후 저장합니다.    

### 2) 장고 서버에 SNS(카카오) 등록 및 REST API 설정을 합니다.

장고 서버의 설정은 https://django-rest-auth.readthedocs.io/en/latest/installation.html 을 따라하면 됩니다.

#### 2-1) rest-auth 설치

+ 먼저 패키지를 설정 합니다.

```python
pip install django-rest-auth
```

<br>

+ `rest_auth`를 project/settings.py의 INSTALLED_APP에 추가 합니다.

```python
INSTALLED_APPS = (
    ...,
    'rest_framework',
    'rest_framework.authtoken',
    ...,
    'rest_auth'
)
```

<br>

참고로 `django-rest-auth`를 수행하기 위해서는 당연히 `django-rest-framework` 환경 설치가 선 진행 되어야 합니다.
따라서 다음을 먼저 설치해 주었는지 확인합니다.

```python
pip install djangorestframework
```

<br>

+ `rest_auth`를 project/urls.py에 등록 합니다.

```python
urlpatterns = [
    ...,
    url(r'^rest-auth/', include('rest_auth.urls'))
]
``` 

<br>

+ 만약 새로 만든 프로젝트라면 `migrate` 를 한번 해줍니다.

```python
python manage.py migrate
```

<br>

#### 2-2) django-allauth 설치 및 Registration 

+ 이제 `django-allauth` 라는 것을 설치해 주어야 합니다. 설치할게 많은데요... 이것은 SNS 계정과 연동하기 위한 패키지 입니다. <br>
  아래 명령어를 통해 설치하시면 됩니다.

```python
pip install "django-rest-auth[with_social]"
```

<br>

+ 패키지를 설치를 하였으니 project/settings.py 의 **INSTALLED_APPS** 에 아래 내용을 등록해 줍니다.

```python
INSTALLED_APPS = (
    ...,
    'django.contrib.sites',
    'allauth',
    'allauth.account',
    'rest_auth.registration',
)
```

<br>

`django.contrib.sites`는 `Multi sites` 기능을 지원해 주는 것을 뜻합니다. 
간단하게 설명하면  Multi sites는 하나의 사이트에서 여러 domain을 가질 수 있는 기능을 말합니다.
Model을 설정할 때, site의 ID를 ForeignKey로 설정할 수 있습니다. 예를 들면 다음과 같습니다.

```python
class Post(models.Model):
    site = models.ForeignKey(SITE_ID)
    ...
``` 

<br>

이렇게 **ForeignKey를 설정해 주면 하나의 모델이지만 여러 서비스에서 사용**될 수 있습니다.
기본적으로 `django-allauth` 는 multi sites 기능을 통하여 구현되기 때문에 INSTALLED_APPS에 `django.contrib.sites`를 입력해 주어야 합니다.
하지만 저희는 여러 사이트를 운영하는 것은 아니고 한 개의 사이트만 운영할 것이기 때문에 Site 아이디를 강제로 지정해주어야 합니다.

따라서 project/settings.py에 다음과 같이 SITE_ID를 강제로 지정합니다. 

```python
SITE_ID = 1
```

<br>

+ project/urls.py 에서 rest-auth/registration/을 등록합니다.

```python
urlpatterns = [
    ...,
    url(r'^rest-auth/', include('rest_auth.urls')),
    url(r'^rest-auth/registration/', include('rest_auth.registration.urls'))
]
```

<br>

#### 2-3) Social Authentication 등록

이제 INSTALLED_APPS에 `allauth.socialaccount`, `allauth.socialaccount.providers.facebook` 을 추가합니다.
앞의 과정을 계속 따라오셨다면 INSTALLED_APPS는 다음과 같습니다.

```python
INSTALLED_APPS = [
    ...,
    'rest_framework',
    'rest_framework.authtoken',
    'rest_auth'
    ...,
    'django.contrib.sites',
    'allauth',
    'allauth.account',
    'rest_auth.registration',
    ...,
    'allauth.socialaccount',
    'allauth.socialaccount.providers.kakao',
]
```

<br>

만약 kakao가 아니라 facebook을 등록하고 싶으시면 `allauth.socialaccount.providers.kakao` 에서 kakao만 facebook으로 바꾸면 됩니다.

#### 2-4) SNS 등록

SNS 계정을 등록할 app을 하나 만들어서 url을 관리해 보도록 하겠습니다. 
+ accounts 라는 앱을 만들고 project/urls.py에 등록 합니다.

```python
# project/urls.py

urlpatterns = [
    ...
    url(r'^accounts/', include('accounts.urls')),
    ...
]
```

<br>

+ accounts/urls.py를 만들고 아래와 같이 입력 합니다.

```python
# accounts/urls.py

from django.conf.urls import include, url
from .views import KakaoLogin

app_name = 'accounts'

urlpatterns = [
    url(r'^rest-auth/kakao/$', KakaoLogin.as_view(), name='kakao_login'),
]
```

<br>

+ accounts/views.py를 만들고 아래와 같이 입력 합니다.

```python

from allauth.socialaccount.providers.kakao.views import KakaoOAuth2Adapter
from rest_auth.registration.views import SocialLoginView

class KakaoLogin(SocialLoginView):
    adapter_class = KakaoOAuth2Adapter
    
```

<br>

#### 2-5) JWT 세팅

`django-rest-auth`는 기본적으로 Django의 Token 기반 인증을 수행합니다.
여기서 JWT 인증을 적용할 것이기 때문에 아래 내용을 실행해야 합니다.

+ [django-rest-framework-jwt](http://getblimp.github.io/django-rest-framework-jwt/) 를 설치합니다.

```python
pip install djangorestframework-jwt
```

<br>

+ `rest-framework`에서 사용할 인증 방식을 JWT로 정하기 위하여 project/settings.py에서 아래와 같이 옵션을 지정해 줍니다.

```python
REST_FRAMEWORK = {

    'DEFAULT_AUTHENTICATION_CLASSES': [
         ...         
         'rest_framework_jwt.authentication.JSONWebTokenAuthentication',
    ],    

}
``` 

<br>

+ 이제 rest-framework에서 Token 발급 시 JWT를 기본으로 사용하도록 명시하겠습니다. project/settings.py에 아래 내용을 추가합니다.

```python
REST_USE_JWT = True
```

<br>

#### 2-6) django admin 에서 세팅

자 이제 다했습니다. 그럼 구현한 내용을 보기 위해서 admin 사이트 (localhost:8000/admin)에 들어가보겠습니다.

![admin1](../assets/img/python/rest/JWT/admin1.PNG)

<br>

admin 접속 후 SITES 항목에 들어가면 `example.com`이 한 개 생성되어 있습니다.
여기를 들어가 보겠습니다. 

![admin2](../assets/img/python/rest/JWT/admin2.PNG)

<br>

URL을 보면 `http://localhost:8000/admin/sites/site/1/change/` 에서 1을 볼 수가 있습니다.
여기서 숫자 1이 저희가 지정한 `SITE_ID = 1`에 해당합니다.

이제 해야할 마지막 작업은 SNS를 admin에 등록해야 합니다. 처음에 만든 카카오 개발자 홈페이지에서
네이티브 앱 키, REST API 키 등이 있었습니다. 이 중 REST API 키를 Django 서버에서 사용한다고 하였는데 이 것을 등록해야 합니다.

+ Django admin의 Social Accounts › Social applications › Add social application 를 클릭합니다.

![admin3](../assets/img/python/rest/JWT/admin3.PNG)

<br>

위와 같이 셋팅을 해주고 등록을 하면 드디어 카카오가 등록 되었습니다.

앱에서 로그인 하기 전에 실제로 서버가 잘 동작하는지 확인하기 위해 access_token을 이용해 보겠습니다.

![kakao5](../assets/img/python/rest/JWT/kakao5.PNG)

<br>

카카오 개발자 페이지에서 `개발가이드 >> REST API` 도구에 들어가서 앱을 선택하고 로그인 하면 Access Token이 생성 됩니다.

 
### 3) 앱을 수정합니다.

+ baseURL 주소를 장고 서버에 맞게 설정 합니다.
+ android-app/app/build.gradle 경로의 applicationId에 유일한 아이디를 할당합니다.
    + 경로 내의 android { defaultConfig { `applicationId` ... } } 
    + applicationId는 앱 별로 유니크 하며 playstore에 한번 올리면 변경할 수 없습니다.
+ android-app/app/src/main/res/values/strings.xml 에서 `app_name` 과 `kakao_app_key`를 변경합니다.
    + app_name에는 사용자가 지정하고 싶은 앱 이름을 넣으면 됩니다.
    + kakao_app_key 에는 `네이티브 앱 키`를 지정하면 됩니다.

```python
<resources>
    <string name = "app_name"> App with kakao </string>
    <string name = "kakao_app_key"> 99ca10c5dd3b1da637b37519b2cc9493 </string>   
</resources>
```

<br>

다시 한번 참조하면 `네이티브 앱` 키는 아래에서 확인할 수 있습니다.

![1](../assets/img/python/rest/JWT/kakao1.PNG) 

<br>

+ 안드로이드 앱 빌드 & 실행을 통해, `keyHash` 값 확인/복사 합니다. 디버깅 창에서 확인할 수 있습니다.

![3](../assets/img/python/rest/JWT/kakao3.PNG)

패키지명에는 위에서 설정한 `applicationId`를 넣습니다. 마켓 URL은 자동 설정 됩니다.

![4](../assets/img/python/rest/JWT/kakao4.PNG)

안드로이드 스튜디오에서 확인한 keyHash 값을 사이트에 등록하면 됩니다.