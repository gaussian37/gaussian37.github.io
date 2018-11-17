---
layout: post
title: Colab with Google Drive 
date: 2018-11-17 11:50:00
img: python/etc/colab-google-drive/colab.png
categories: [python-etc] 
tags: [python, colab, google drive] # add tag
---

This blog is about how to use google-drive files in `colab`.

It's simple.

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

And **Go to this URL in a browser** and enter your authorization code: <br>

Then, you can access your drive with following path.

```python
gdrive/My Drive/
```