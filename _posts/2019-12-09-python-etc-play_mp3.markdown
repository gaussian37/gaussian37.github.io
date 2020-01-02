---
layout: post
title: pygame으로 mp3 출력
date: 2019-12-09 00:00:00
img: python/etc/pygame/0.png
categories: [python-etc] 
tags: [python, pygame] # add tag
---

<br>

- 파이썬에서 mp3 파일을 실행시키는 방법은 다양하게 있습니다.
- 그 중 음질 손실, lagging, 노이즈 문제에 다소 강건한 라이브러리인 `pygame`을 이용하여 mp3를 출력하는 방법에 대하여 정리하였습니다.

<br>

- 설치 : `pip install pygame`

<br>

```python
import pygame
# 초기화 시 아래와 같이 입력하면 lagging 문제를 해결할 수 있다고 한다.
pygame.mixer.init(48000, -16, 1, 1024)
pygame.mixer.music.load("music.mp3")
pygame.mixer.music.play()
...
pygame.mixer.music.stop()
```

<br>

- `pygame`에서 mp3를 재생하게 되면 background에서 돌아가게 됩니다.
- 따라서 play()와 stop()사이에 어떤 작업을 처리할 수 있습니다.