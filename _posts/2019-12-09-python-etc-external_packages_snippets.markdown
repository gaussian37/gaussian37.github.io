---
layout: post
title: python 외부 package snippets
date: 2019-12-09 00:00:00
img: python/etc/external_packages/0.png
categories: [python-etc] 
tags: [python, external packages, snippets] # add tag
---
<br>

## **목차**

<br>

- ### pygame 으로 mp3 파일 출력
- ### pyautogui 으로 화면 해상도 출력
- ### sourcedefender를 이용한 소스코드 암호화
- ### moviepy를 이용한 동영상 붙이기

<br>

## **pygame 으로 mp3 파일 출력**

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

<br>

## **pyautogui 으로 화면 해상도 출력**

<br>

- 파이썬 코드를 이용하여 화면 해상도를 확인하고 싶을 때 `pyautogui`를 이용할 수 있습니다.

<br>

```python
import pyautogui
width_resolution, height_resolution = pyautogui.size()
```

<br>

## **sourcedefender를 이용한 소스코드 암호화**

<br>

- 파이썬 소스코드를 암호화하여 전달이 필요할 때, `sourcedefender`를 사용하여 암호화 할 수 있습니다.
- 설치 : `pip install sourcedefender`
- 암호화 방법 : `sourcedefender encrypt 파일.py`, 암호화를 하면 `파일.pye`라는 파일이 생성되며 이 파일은 읽었을 때, 암호화가 되어 있어서 사람이 읽을 수 없게 되어 있습니다.
- 암호화 파일 실행 방법 : `python -m sourcedefender 파일.pye`. 실행하는 측에서도 `sourcedefender`는 설치가 되어 있어야 `.pye` 파일을 실행할 수 있습니다.
- 암호화된 파일은 다시 복호화 할 수 없기 때문에 (sourcedefender에서 기능을 제공하지 않음) 안전하게 암호화 해서 코드에 사용된 알고리즘을 비공개 처리 할 수 있습니다.

<br>

## **moviepy를 이용한 동영상 붙이기**

<br>

- 2개 이상의 동영상을 차례 대로 붙여서 하나의 동영상으로 만드는 방법을 `moviepy`를 이용하면 간단하게 구현할 수 있습니다.

<br>

``` python
from moviepy.editor import VideoFileClip, concatenate_videoclips

clip_1 = VideoFileClip("video1.mp4")
clip_2 = VideoFileClip("video2.mp4")
final_clip = concatenate_videoclips([clip_1,clip_2])
final_clip.write_videofile("final.mp4")
```

<br>

- 만약 특정 path의 `*.mp4` 파일을 이름순으로 차례대로 이어 붙이려면 아래 코드를 사용하면 됩니다.

<br>

```python
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

video_path = "./"
video_names = os.listdir(video_path)
video_names.sort()
clips = [VideoFileClip(video_name) for video_name in video_names if video_name.split(".")[-1] == "mp4"]
final_clip = concatenate_videoclips(clips)
final_clip.write_videofile("final.mp4")
```

<br>