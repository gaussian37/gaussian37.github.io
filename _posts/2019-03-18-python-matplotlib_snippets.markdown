---
layout: post
title: Matplotlib 기본 문법 및 코드 snippets
date: 2019-03-18 00:00:00
img: python/basic/matplotlib/0.png
categories: [python-basic] 
tags: [Matplotlib, python, python 기본] # add tag
---

<br>

- 이 글에서는 `Matplotlib`를 사용하면서 필요하다고 느끼는 `Matplotlib 기본 문법 및 코드`들을 정리해 보겠습니다.

<br>

## **목차**

<br>

- ### matplotlib에서 한글 사용 하기

<br>

- matplotlib를 이용하여 그래프를 그릴 때, 한글을 사용해야 한다면 반드시 한글을 지원하는 폰트를 사용하도록 지정해 주어야 합니다.
- 아래 코드를 지정하면 한글 폰트인 맑은 고딕체를 이용하여 matplotlib에 텍스트를 출력합니다.

<br>

```python
import matplotlib.font_manager as fm
from matplotlib import rc
font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
```

<br>
