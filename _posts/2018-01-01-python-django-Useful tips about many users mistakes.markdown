---
layout: post
title: Useful tips about many users mistakes
date: 2018-01-01 18:43:00
img: python/django/django.png
categories: [python-django] 
tags: [python, django, tips, mistakes] # add tag
---

# `app_name` error

`app_name` error happens from django 2.0 or over. 
Before 2.0 ver, urlpattens of project urls.py, you just added `namespace` in the url().

```python
# project_name/urls.py
urlpatterns = [
    ....
    url(r'^sample/', include('sample.urls', namespace='sample'))
]
```  

<br>

And you don't need to add `app_name` in other file.
but from django ver 2.0, you must add `app_name` in application urls.py.
For instance, according to example above, you must add following line in sample/urls.py (sample is application name).

```python
# sample/urls.py

app_name = 'sample'
```

<br>

