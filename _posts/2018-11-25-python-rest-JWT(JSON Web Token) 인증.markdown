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
    
### Token은 반드시 안전한 장소에 보관해야 합니다.

+ 일반 Token / JWT 토큰 여부에 상관 없이 Token은 반드시 안전하게 보관 되어야 합니다. (인증 관련 요소이므로)
+ 스마트폰 앱은, 설치된 앱 별로 안전한 저장 공간이 제공되므로 토큰 방식을 쓰는 것이 좋지만 웹브라우저를 사용할 때는 저장 공간이 없습니다.
    + **Token은 앱 환경**에서만 권장 됩니다.
    + **웹 클라이언트 환경**에서는 보안적인 측면에서 **세션 인증**이 나은 선택일 수도 있습니다.  
    
   
### Token 예시

8df73dafbde4c669dc37a9ea7620434515b2cc43

### JWT 예시

eyJ0eXAi...IUzI1NiJ9`.`eyJ1c2VyX2lkIjo...aWwiOiIifQ`.`Zf_o3S7Q7-cmUz...LcF-2VdokJQ (너무 길어서 ... 로 줄였습니다.)

+ JWT는 **.** 을 기준으로 3영역으로 나뉘게 됩니다.
+ 헤더(Header)를 base64 인코딩하여, eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9
+ 내용(Payload)를 base64 인코딩하여, eyJ1c2VyX2lkIjoxLCJ1c2VybmFtZSI6ImFza2RqYW5nbyIsImV4cCI6MTUxNTcyMTIxMSwiZW1haWwiOiIifQ
+ 서명(Signature) : Header/Payload를 조합하고 비밀키로 서명한 후 base64 인코딩하여, Zf_o3S7Q7-cmUzLWlGEQE5s6XoMguf8SLcF-2VdokJQ

### DRF에 JWT 세팅하기

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

### HTTPie를 통한 JWT 발급

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

### 발급받은 JWT Token 확인

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

### 발급받은 JWT Token으로 포스팅 목록 API 요청

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

### JWT Token 유효기간이 지났을 경우

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

### JWT Token 갱신받기

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


### djangorestframework-jwt의 주요 settings

```python
JWT_AUTH = { 
    'JWT_SECRET_KEY': settings.SECRET_KEY, 
    'JWT_ALGORITHM': 'HS256', 
    'JWT_EXPIRATION_DELTA': datetime.timedelta(seconds=300), 
    'JWT_ALLOW_REFRESH': False, 
    'JWT_REFRESH_EXPIRATION_DELTA': datetime.timedelta(days=7), }
```

<br>

기본적인 Setting 값은 위와 같습니다. 여기서 바꿔줘야 하는 것은 `JWT_EXPIRATION_DELTA`, `JWT_ALLOW_REFRESH`, `JWT_REFRESH_EXPIRATION_DELTA` 항목 입니다.
+ `JWT_EXPIRATION_DELTA` : Token 만료시간으로 기본값은 5분 입니다. 5분은 너무 짧으니 늘려 줍시다.
+ `JWT_ALLOW_REFRESH` : Token Refresh가 가능하도록 해줘야 합니다. 따라서 True로 바꾸어 줍니다.
+ `JWT_REFRESH_EXPIRATION_DELTA` : Refresh 가능 시간 입니다. 상식적으로 만료 되기전에 Refresh를 하는게 맞으므로 위의 `JWT_EXPIRATION_DELTA` 시간보다 짧게 해주면 됩니다.


## 실제 SNS랑 연동해서 사용해 보려면?

카카오를 예를 들어 설명해 보겠습니다. 카카오로부터 Access Token을 획득하고, 
이를 장고 서버를 통해 JWT 토큰을 획득해야 합니다.

+ 앱↔카카오톡 서버
    + 안드로이드 앱에서 "카카오톡 로그인" 버튼을 클릭하면, 카카오톡 서버와 인증을 수행합니다.
    + 카카오톡 서버와의 인증에 성공하면, 카카오톡으로부터 Access Token 을 획득

+ 앱↔장고 서버
    + 획득된 Access Token을 장고 서버 인증 Endpoint (/accounts/rest-auth/kakao/)를 통해, JWT 토큰 획득
        + /accounts/rest-auth/kakao/ 는 django 서버에서 따로 설정해 주어야 하는 URL 입니다.
        + kakao 말고 따른 SNS 를 추가 하고 싶으면 /accounts/rest-auth/facebook/ 과 같이 설정할 수 있도록 유연하게 설계하면 됩니다.
    + 획득한 JWT 토큰이 만료되기 전에, `갱신` 합니다.
    + 획득한 JWT 토큰이 만료되었다면, Access Token을 서버로 전송하여 `JWT 토큰 재획득`