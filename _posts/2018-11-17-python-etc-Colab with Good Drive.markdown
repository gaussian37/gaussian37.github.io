---
layout: post
title: Colab with Google Drive 
date: 2018-11-17 11:50:00
img: python/etc/colab-google-drive/colab.png
categories: [python-etc] 
tags: [python, colab, google drive] # add tag
---

colab에서 어떻게 google drive 와 연동해서 쓰는 지 알려드리겠습니다. 내용은 아래와 같이 간단합니다.

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

<br>

위 코드를 Jupyter notebook에서 실행하면 인증을 위한 URL 링크가 생성 됩니다.
URL 링크를 따라 들어가서 인증 키를 colab에 입력 시키면 아래와 같은 디렉토리가 접근이 가능해 집니다.

```python
gdrive/My Drive/
```

<br>

이제 `My Drive` 아래에 접근하면 본인이 인증한 계정에 해당하는 구글 드라이브 내의 파일이 접근 가능합니다.

자! 이제 즐거운 코딩을 해봅시다!!