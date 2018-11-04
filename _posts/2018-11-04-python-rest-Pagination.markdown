---
layout: post
title: DRF Pagination
date: 2018-11-04 18:43:00
img: python/rest/rest.png
categories: [python-rest] 
tags: [python, django, rest, pagination] # add tag
---

### What is pagination?

If there are too many records in a list, it's better to avoid requesting only one API.
In this case, you can seperate the requests in several pages. And DRF(Django RestFramework) support this `pagination`.

+ PageNumberPagination : Pagination with `page` parameters 
    - http://api.example.org/accounts/?page=4
    - http://api.example.org/accounts/?page=4&page_size=100
+ LimitOffsetPagination : Pagination with `limit` parameter
    - http://api.example.org/accounts/?limit=100
    - http://api.example.org/accounts/?offset=400&limit=100

In `rest_framework/generics.py`, `GeneericAPIView` has `pagination_class =  PageNumberPagination` setting.
But from Ver. 3.7.0, this setting has been change to `pagination_class =  None`.

Accordingly, There is no pagination process because `PAGE_SIZE` is `None`.

If you want to apply `pagination`, there are two options.

+ apply pagination `globally`

In `Project/settings.py`, set option below. set the specific `PAGE_SIZE`.  

```python
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS':'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE':20
}
```

<br>

+ apply pagination at `API`

```python
# application/pagination.py
from rest_framework.pagination import PageNumberPagination

class PostPageNumberPagination(PageNumberPagination):
    page_size = 20
    
# application/views.py
from .pagination import PostPageNumberPagination

class PostViewSet(viewsets.ModelViewSet):
    ...
    pagination_class = RestaurantPageNumberPagination
```

<br>

You can set default pagination in `globally` and additionally set another pagination `API by API`.
