---
layout: post
title: django queryset (장고 쿼리셋) 관련 내용
date: 2019-01-20 18:43:00
img: python/django/django.png
categories: [python-django] 
tags: [python, django, queryset, 장고, 쿼리셋] # add tag
---

django를 사용하다 보면 queryset 관련하여 명령어가 상당히 혼란스러울 때가 많은 것 같습니다.

이 글에서는 개인적으로 프로젝트를 진행하다가 참조할 겸 정리할 겸 queryset 내용들을 정리해 보겠습니다.

<br>

## filter 

<br>

+ 장고에서 테이블을 조회할 때, `filter`를 사용하여 필요한 데이터만 조회하는 작업을 합니다.
+ `filter`를 사용하여 데이터를 `lookup`하는 대표적인 방법 입니다.

<br>

+ `exact` : 정확히 일치하는 데이터 찾기
    + 만약 `None`을 찾는다고 하면 `Null`을 찾는 것과 동일합니다. **isnull**

```python
Entry.objects.get(id__exact=14)
Entry.objects.get(id__exact=None)
``` 

<br>

+ `iexact` : 대소문자를 구분하지 않고 정확히 일치하는 데이터 찾기

```python
Blog.objects.get(name__iexact='beatles blog')
Blog.objects.get(name__iexact=None)
```

<br>

+ `contains`, `icontains` : 포함하는 문자열 찾기 (`icontains`는 대소문자 구분하지 않음)
+ 아래 코드는 headline에서 **Lennon**이라는 문자열을 포함하는 object를 찾습니다.

```python
Entry.objects.get(headline__contains='Lennon')
```

<br>

+ `in` : list, tuple, string 또는 queryset과 같이 iterable한 객체를 대상으로 각 원소를 조회합니다.

```python
Entry.objects.filter(id__in=[1, 3, 4])
: SELECT ... WHERE id IN (1, 3, 4);

Entry.objects.filter(headline__in='abc')
: SELECT ... WHERE headline IN ('a', 'b', 'c');
``` 

<br>

+ 또는 다음과 같이 queryset를 직접 조건으로 넣을 수 있습니다. (성능 체크 필요)

```python
inner_qs = Blog.objects.filter(name__contains='Cheddar')
entries = Entry.objects.filter(blog__in=inner_qs)
:SELECT ... WHERE blog.id IN (SELECT id FROM ... WHERE NAME LIKE '%Cheddar%')
```

<br>

+ `gt`, `gte`, `lt`, `lte` 와 같이 부등호를 사용할 수 있습니다.

```python
Entry.objects.filter(id__gt=4)
: SELECT ... WHERE id > 4;
```

<br>

+ `startswith`, `istartswith`, `endswith`, `iendswith`는 각각 접미사, 접두사를 찾습니다.

```python
Entry.objects.filter(headline__startswith='Lennon')
Entry.objects.filter(headline__endswith='Lennon')
```

<br>

+ `range`는 범위에 해당하는 object를 찾습니다.

```python
import datetime
start_date = datetime.date(2005, 1, 1)
end_date = datetime.date(2005, 3, 31)
Entry.objects.filter(pub_date__range=(start_date, end_date))
```
