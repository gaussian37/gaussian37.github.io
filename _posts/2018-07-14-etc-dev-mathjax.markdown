---
layout: post
title: Jekyll 에서 MathJax 사용하는 방법
date: 2018-07-14 18:30:00
img: etc/dev/1.png
tags: [MathJax] # add tag
---

- mathjax 데모 페이지 : https://www.mathjax.org/#demo
- mathjax 레퍼런스 : https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference
- dextify (mathjax 심볼 인식 사이트) : https://detexify.kirelabs.org/classify.html
- Jekyll 에서 Mathjax를 사용하려면 `_layouts` 디렉토리 안의 `main.html`에 아래 코드를 삽입합니다.

<br>

```
<script src="//cdnjs.cloudflare.com/ajax/libs/mathjax/2.5.3/MathJax.js?config=TeX-AMS-MML_SVG"></script>
```

<br>

- 위 코드에서 `2.5.3`은 버전이고 다른 버전을 입력해도 됩니다.
- mathjax를 사용할 때에는 `$$` 로 mathjax 문법을 감싸면 페이지에 적용됩니다.