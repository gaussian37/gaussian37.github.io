---
layout: post
title: DRF의 Authorization이 있을 때, 웹사이트 사용 방법
date: 2019-05-02 00:00:00
img: python/rest/rest.png
categories: [python-rest] 
tags: [python, django, REST, drf, Authorization, 웹사이트] # add tag
---

+ Django의 큰 장점 중 하나는 Admin 페이지를 쉽게 관리할 수 있다는 것입니다.
+ Django RestFramework를 사용할 때에도 Serializer를 이용하면 페이지를 쉽게 관리할 수 있습니다.
+ 이 때, 만약 Token 또는 JWT 를 사용하여 권한의 제한을 두었다면 DRF 페이지를 어떻게 사용할 수 있을까요?

<br>

+ 먼저 https://gaussian37.github.io/python-rest-JWT-Authorization/ 를 이용하여 API 중 `api-jwt-auth` 구조를 만들어야 합니다.
    + 즉, localhost:8000/api-jwt-auth/ 구조를 위 블로그 링크를 보고 만들어야 합니다.
+ 다음으로 `python manage.py createsuperuser`를 통하여 관리자 계정을 만들어야 합니다.
+ `api-jwt-auth`를 만들고 `superuser` 계정을 이용하면 관리자 계장에 따른 JWT를 얻을 수 있습니다. 그러면 환경 구축을 한 가정으로 설명을 진행하겠습니다.

<br>

### Restlet Client를 이용한 JWT 획득

+ 먼저 JWT를 획득하기 위해서는 API에 POST를 하여 JWT를 얻어야 합니다.
    + API 호출 주소는 다음과 같습니다. (localhost:8000 부분은 실제 사용하는 url 주소를 입력하면 됩니다.)
        + API : https://localhost:8000/api-jwt-auth/
        + Body : {"username":유저ID, "password":비밀번호} 
+ 만약 API를 호출할 수 있는 환경이면 위 API주소에 Body를 넣어서 보내면 JWT를 받을 수 있습니다.
+ 만약 API를 호출할 수 없는 환경이면 크롬에서 `Restlet Clinet`를 설치하여 쉽게 JWT를 얻을 수 있습니다.

<img src="../assets/img/python/rest/drf-website/restlet.PNG" alt="Drawing" style="width: 600px;"/>

+ 먼저 크롬 extention에서 `restlet`을 설치합니다. 

<img src="../assets/img/python/rest/drf-website/restlet-ex.PNG" alt="Drawing" style="width: 1000px;"/>
        
+ restlet을 실행하면 위와 같은 화면이 실행됩니다.
+ 이 때, POST로 놓고 API 주소를 입력한 다음, Body에 username과 password를 입력 한뒤 `send`를 누르면 성공적으로 JWT를 얻을 수 있습니다.

<br>

### modheader를 이용한 Authorization 헤더 입력

+ 현재 문제는 DRF 사이트를 접속 하였을 때, `JWT Authorization`의 권한이 없어서 사이트 사용이 불가능한 상황입니다.
+ 그러면 어떻게 Authorization을 줄 수 있는지 알아보겠습니다.

<img src="../assets/img/python/rest/drf-website/modheader.PNG" alt="Drawing" style="width: 600px;"/>

+ 위와 같이 크롬 extension에서 `modheader`를 설치 후 실행합니다.

<img src="../assets/img/python/rest/drf-website/modheader-ex.PNG" alt="Drawing" style="width: 600px;"/>

+ 그 다음 웹사이트에 접속할 때, `Authorization` 헤더를 넘겨주기 위해 header 정보를 입력 합니다.
    + header에는 `Authorization`을 입력합니다.
    + value에는 `JWT 실제토큰값`을 입력합니다.(이 때, JWT를 먼저 쓰고 한칸 띄운 다음 실제 토큰값을 넣으면 됩니다.)
+ 위와 같이 입력을 하면 JWT 토큰을 헤더로 넘기기 때문에 인증이 완료 됩니다.