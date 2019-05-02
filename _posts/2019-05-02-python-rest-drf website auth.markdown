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

### Restlet Client를 이용한 JWT 획득

+ 먼저 https://gaussian37.github.io/python-rest-JWT(JSON-Web-Token)-%EC%9D%B8%EC%A6%9D/
+ 먼저 JWT를 획득하기 위해서는 API에 POST를 하여 JWT를 얻어야 합니다.
    + 호출 주소 


