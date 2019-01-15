---
layout: post
title: 장고 쉘(Django shell)
date: 2018-09-16 18:43:00
img: python/django/django.png
categories: [python-django] 
tags: [python, django, shell, shell_plus, django_extensions] # add tag
---

## What is django shell?

+ It's kind of python shell which has django project setting
    - python manage.py shell
+ Basic python shell can not access django environment
    - 그냥 python 또는 ipython을 실행하면 환경 변수가 등록 되지 않아 현재 프로젝트에 접근할 수 없습니다.
    - **project setting** is not loaded
    - **project setting** is about project/settings.py
+ Tip : If ipython is installed, shell would be run ipython
    - Basically, ipython is installed with jupyter notebook
    - In order to install ipython <br>
        ```python
        python -m pip install --upgrade pip
        pip install "ipython[notebook]" 
        ```
+ 장고 쉘 실행 시 우선순위
    + 1) ipython, 2) bpython 3) python
    + 옵션
        + -i 또는 (--interface) : 인터프리터 인터페이스 커스텀 지정
        + -c 또는 (--command) : 실행할 파이썬 코드를 문자열로 지정
        
        
<br><br>

## Run with shell

```python
python manage.py shell
```

<br><br>
        
## shell_plus and jupyter notebook with django shell

+ Install django-extensions
    - `pip install django-extensions`
    - add `'django_extensions'` in project/settings.py : `INSTALLED_APPS` = [  ]
    - 주의할 점은 `pip install` 시에는 `하이픈(-)`을 사용하여 django-extensions 이고 `INSTALLED_APPS 추가` 시에는 `언더바(_)` django_extensions 입니다.
+ run with shell_plus
    - python manage.py `shell_plus` : shell + pre-imported packages    
        - ``` python manage.py shell_plus ```
    - python manage.py `shell_plus --notebook` : shll_plus with jupyter notebook
        - ``` python manage.py shell_plus --notebook```
        
<br><br>
    
## If error occurs...

CommandError: No notebook (Python) kernel specs found

In the shell_plus, computer find the kernel which has name of `python` or `python3`. But if not, This CommandError occurs. <br>
First, check the available kernel list

```python
from nb_conda_kernels import manager 
kernel_specs = manager.CondaKernelSpecManager().find_kernel_specs() 
print(kernel_specs) # ex) ['Python [Root]']
```

<br><br>

In the project/settings.py <br>
set the kernel list in the settings.NOTEBOOK_KERNEL_SPEC_NAMES

```python
NOTEBOOK_KERNEL_SPEC_NAMES = ['Python [Root]']
```

<br><br>

## SQL 출력 옵션

shell에서 SQL을 출력하려면 다음과 같이 실행하면 되겠습니다.

```python
python manage.py shell_plus --print-sql
```

<br>

또는

<br>

```python
settings.SHELL_PLUS_PRINT_SQL = True
```

<br><br>

## 기본 shell에서 장고 기능을 사용하고 싶으면 ...

script를 작성한 후 shell에 넘겨서 사용하려면 다음 코드를 입력해 주면 됩니다.

But It's inconvenient, Let's use django `shell_plus`
 
But if you want to use it, refer below,

```python
import os
import django 
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproj.settings') # FIXME: check path
django.setup()
```

<br><br>

## 기본 shell에서 장고 shell_plus 처럼 모든 패키지를 등록해서 사용

기본 shell에서 장고 shell_plus 처럼 사용하고 싶으면 script에 아래와 같이 등록합니다.

+ 먼저 위에서 설명한 장고의 환경 변수를 등록해 줍니다.

```python
import os
import django 
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproj.settings') # FIXME: check path
django.setup()
```

<br>

+ shell_plus를 실행하였을 때 실행되는 패키지들을 script에 등록해 줍니다.
+ 아래 내용은 제 컴퓨터 환경에서 실행시켰을 때, 생성되는 내용의 예 입니다.

```python
# Shell Plus Model Imports
from allauth.account.models import EmailAddress, EmailConfirmation
from allauth.socialaccount.models import SocialAccount, SocialApp, SocialToken
from dining.models import Image, Like, Restaurant, Review
from django.contrib.auth.models import Group, Permission, User
from django.contrib.admin.models import LogEntry
from django.contrib.contenttypes.models import ContentType
from django.contrib.sessions.models import Session
from django.contrib.sites.models import Site
from rest_framework.authtoken.models import Token
# Shell Plus Django Imports
from django.core.cache import cache
from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import transaction
from django.db.models import Avg, Case, Count, F, Max, Min, Prefetch, Q, Sum, When, Exists, OuterRef, Subquery
from django.utils import timezone
from django.urls import reverse

```