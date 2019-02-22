---
layout: post
title: Pythonanywhere 세팅 관련 모음
date: 2019-02-21 13:46:00
img: python/etc/pythonanywhere/pythonanywhere.PNG
categories: [python-etc] 
tags: [python, PaaS, pythonanywhere] # add tag
---

## Web Tab에서 필요값 설정

+ Pythonanywhere에서 Django로 서버를 배포하기 위해서는 아래 세가지를 세팅해야 합니다.
    + Working directory (아래 빨간색 상자)
        + Django 프로젝트가 있는 디렉토리를 설정해 주어야 합니다
        + 아래 예에서는 `djangoserver`라는 디렉토리가 프로젝트의 시작 디렉토리입니다.
    + WSGI configuration file은 클릭을 한 다음에 안에 내부 값을 수정해야 합니다. (아래 파란색 상자)
        + ```python
          import os
          import sys
            
          # path = '/home/USER_NAME/PROJECT'
          path = '/home/gaussian37/djangoserver'
            
          if path not in sys.path:
              sys.path.append(path)
            
          # os.environ['DJANGO_SETTINGS_MODULE'] = 'mysite.settings'
          os.environ['DJANGO_SETTINGS_MODULE'] = 'djangoserver.settings'
            
          from django.core.wsgi import get_wsgi_application
          from django.contrib.staticfiles.handlers import StaticFilesHandler
          application = StaticFilesHandler(get_wsgi_application()) 
          ```
    + virtualenv 경로를 설정해야 합니다. (아래 초록색 상자)
        + virtualenv 경로를 지정해 줍니다.
        + 예를 들어 ``` /home/gaussian37/djangoserver/venv ``` 라고 설정한 경우는 project인 djangoserver 내부에 venv라는 가상환경이 있는 경우 입니다.

<img src="../assets/img/python/etc/pythonanywhere/websetting.PNG" alt="Drawing" style="width: 600px;"/>


<br><br>

## MySQL 세팅하기

+ 서버를 최종 배포할 때에는 SQLite가 아닌 MySQL 또는 Postgres 등을 사용해야 합니다.
+ MySQL을 사용하는 방법은 상당히 간단하니 아래 내용을 참조하시면 됩니다.

<img src="../assets/img/python/etc/pythonanywhere/databases.PNG" alt="Drawing" style="width: 600px;"/>

+ 먼저 `Databases` 탭에 들어갑니다.
+ 이 때 확인해야 할 정보들은 다음과 같습니다.
    + Conneting
        + Database host address : `HOST` 라는 항목에 사용할 주소입니다.
        + `Username`
    + Your databases
        + 사용가능한 DB Table의 목록 입니다.
        + `USER_NAME$DB_NAME` 형태로 되어 있습니다.
        + 각 DB Table을 누르면 MySQL 콘솔창이 실행됩니다.
    + Create database
        + DB Table을 생성합니다.
        
<br>

+ MySQL 콘솔에 접근하는 방법은 2가지가 있습니다.
    + MySQL Setting의 Your databases 목록에서 DB Console 클릭하여 실행하기
    + 일반 bash에서 다음 명령어로 실행하기
        + ``` mysql -u USERNAME -h HOSTNAME -p 'USERNAME$DATABASENAME' ```
        
<br>

+ pythonanywhere의 파이썬에서 MySQL을 사용하기 위해서
    + virtualenv를 사용하지 않는 경우
        + ``` import MySQLdb ``` 입력
    + virtualenv를 사용하는 경우
        + python 2.X
            + ``` pip install mysql-python ```
        + python 3.x
            + ``` pip install mysqlclient ```

<br>

+ Django에 MySQL 설정하려면 앞에서 사용하였던 databases 탭의 정보를 가져와야 합니다.
+ 아래 코드는 project/settings.py의 DATABASES 값에 해당하고 아래와 같이 수정해야 합니다.
    + 기본 값은 sqlite가 지정되어 있습니다.

<img src="../assets/img/python/etc/pythonanywhere/databases.PNG" alt="Drawing" style="width: 600px;"/>

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': '<your_username>$<your_database_name>',
        'USER': '<your_username>',
        'PASSWORD': '<your_mysql_password>',
        'HOST': '<your_mysql_hostname>',
    }
}
```

+ 위 이미지와 코드를 조합해 보면 예제는 다음과 같습니다.

```python

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'gaussian37$dining',
        'USER': 'gaussian37',
        'PASSWORD': '비밀번호',
        'HOST': 'gaussian37.mysql.pythonanywhere-services.com',
    }
}

```