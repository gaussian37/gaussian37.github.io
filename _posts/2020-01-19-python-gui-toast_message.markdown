---
layout: post
title: Python을 이용한 Toast Message 생성
date: 2019-12-09 00:00:00
img: python/gui/gui.png
categories: [python-gui] 
tags: [python, toast message] # add tag
---

<br>

- Python을 이용하여 토스트 메시지를 생성하는 방법이 다양하게 있는데 그 중 `plyer` 라이브러리를 사용하는 방법에 대하여 알아보도록 하겠습니다.
- `plyer` 라이브러리의 장점은 여러 OS를 지원하기 때문에 코드 재사용성이 좋은 것에 있습니다.
- `plyer` 라이브러리 참조 : https://plyer.readthedocs.io/en/latest/#

<br>

- 설치 방법 : `pip install plyer`
- 사용 방법 :

<br>
<center><img src="../assets/img/python/gui/plyer/0.png" alt="Drawing" style="width: 800px;"/></center>
<br> 

```python
from plyer import notification

notification.notify(
    title = '제목입니다.',
    message = '메시지 내용입니다.',
    app_name = "앱 이름",
    app_icon = 'bluemen_white.ico', # 'C:\\icon_32x32.ico'
    timeout = 10,  # seconds
)
```

<br>
<center><img src="../assets/img/python/gui/plyer/1.png" alt="Drawing" style="width: 700px;"/></center>
<br> 
