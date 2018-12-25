---
layout: post
title: Model and Model Fields
date: 2018-10-28 18:43:00
img: python/django/django.png
categories: [python-django] 
tags: [python, django, related_name, prefetch_related] # add tag
---

오늘은 저처럼 실수하시는 분이 없기 위하여 글을 써보려고 합니다.

장고 model 에서 ForeignKey를 사용 할 때, 특히 한 모델에서 여러 개의 ForeignKey가 있을 때,
related_name을 이용하여 ForeignKey의 이름을 정할 수 있습니다.

예를 들어 모델 A의 pk를 모델 B가 FK로 사용하고 있다고 가정합시다.
장고는 lazy 하게 SQL 작업을 수행하므로 ForeignKey로 접근한 데이터에 대한 작업이 여러번 생긴다면,
즉, **join 연산이 필요**하다면, `prefetch_related`를 통하여 필요한 query를 바로 가져와서 작업하는 것이 효율적 입니다.

이 때, `prefetch_related`는 다음과 같이 사용할 수 있습니다.

```python
A.objects.prefetch_related("B_set")
```

<br>

즉, A 모델이 B 모델의 FK를 통하여 join 한 결과 값을 lazy하지 않게 한번에 가져올 때 사용합니다.
`prefetch_ralated` 내용은 제 블로그의 다른 글에서 확인해 보시면 되겠습니다.

이 때 중요한 것은 related_name을 이용한 경우 `model_set` 형태가 아니라 `related_name` 자체를 입력해 주면 되겠습니다.

예를 들면 아래와 같습니다.

```python
class Price(models.Model):
    book = models.ForeignKey(Book, related_name='prices')
    
books = Book.objects.prefetch_related('prices')
```

<br>

다시 하면 정리하면 `related_name`을 사용한 경우 `prefetch_related`를 사용할 때, `foo_set`을 파라미터로 사용하지 않고
`related_name`을 바로 사용하면 됩니다. 

도움이 되셨다면 광고 클릭해주시면 큰 도움이 되겠습니다. 꾸벅