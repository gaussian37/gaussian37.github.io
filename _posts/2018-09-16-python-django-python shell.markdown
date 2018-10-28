---
layout: post
title: Django shell
date: 2018-09-16 18:43:00
img: python/django/django.png
categories: [python-django] 
tags: [python, django, shell] # add tag
---

### What is django shell?

+ It's kind of python shell which has django project setting
    - python manage.py shell
+ Basic python shell can not access django environment
    - **project setting** is not loaded
    - **project setting** is about project/settings.py
+ Tip : If ipython is installed, shell would be run ipython
    - Basically, ipython is installed with jupyter notebook
    - In order to install ipython <br>
        ```python
        python -m pip install --upgrade pip
        pip install "ipython[notebook]" 
        ```
<br>
        
### shell_plus and jupyter notebook with django shell

+ Install django-extensions
    - pip install django-extensions
    - add `'django_extensions'` in project/settings.py : `INSTALLED_APPS` = [  ]
+ run with shell_plus
    - python manage.py `shell_plus` : shell + pre-imported packages    
    - python manage.py `shell_plus --notebook` : shll_plus with jupyter notebook
    
### If error occurs...

CommandError: No notebook (Python) kernel specs found

In the shell_plus, computer find the kernel which has name of `python` or `python3`. But if not, This CommandError occurs. <br>
First, check the available kernel list

```python
from nb_conda_kernels import manager 
kernel_specs = manager.CondaKernelSpecManager().find_kernel_specs() 
print(kernel_specs) # ex) ['Python [Root]']
```

<br>

In the project/settings.py <br>
set the kernel list in the settings.NOTEBOOK_KERNEL_SPEC_NAMES

```python
NOTEBOOK_KERNEL_SPEC_NAMES = ['Python [Root]']
```

<br>

### In the basic shell, we can also access the django...

But It's inconvenient, Let's use django `shell_plus`
 
But if you want to use it, refer below,

```python
import os
import django 
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproj.settings') # FIXME: check path
django.setup()
```