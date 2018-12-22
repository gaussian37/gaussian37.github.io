---
layout: post
title: queryset 오름차순, 내림차순 정렬
date: 2018-12-22 00:00:00
img: python/django/django.png
categories: [python-django] 
tags: [python, django, Model, Model Field] # add tag
---

장고 ORM을 이용하여 DB를 읽을 때, 기본적으로 필요한 작업이 
오름차순(ascending)/내림차순(descending)으로 특정 field를 가져오는 것입니다.

ORM을 이용하여 DB를 가져올 때 어떻게 하면 될까요?

`A`라는 모델이 있다고 가정합시다. `A`의 모든 데이터를 긁어 오려면 다음과 같이 입력하면 됩니다.

```python
A.objects.all()
```

<br>

그 다음에 `order_by()`를 사용하면 됩니다. 이 때 인자로 들어갈 문자열은 field의 이름입니다.
A 라는 모델에 `point`라는 field가 있다고 합시다. 그러면 다음과 같이 읽어올 수 있습니다.

```python
A.objects.all().order_by('point')
```

<br>

이렇게 읽어오면 오름차순으로 읽어오게 됩니다. 기본값은 오름차순 입니다.

```python
A.objects.all().order_by('-point')
```

<br>

이렇게 읽어오면 내림차순으로 읽어오게 됩니다.

정리하면 `order_by('filed 이름)`으로 DB 조회 시 오름차순/내림차순을 결정하게 되고 이 때,
filed이름 앞에 **-** 를 붙이면 내림차순으로 읽어오게 됩니다.

도움이 되셨으면 광고나 한번 클릭 부탁 드립니다. 꾸벅