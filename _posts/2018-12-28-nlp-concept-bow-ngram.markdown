---
layout: post
title: N-gram과 BoW(Bag-of-Words)
date: 2018-12-28 00:10:00 +0300
img: nlp/concept/bow/bow.png
categories: [nlp-concept] 
tags: [nlp, n-gram, bow, bag of words] # add tag
---

## N-gram과 BoW

N-gram과 BoW는 간단하게 구할 수 있고, 머신러닝 기반의 다소 얕은 학습 방법에서는 효과가 있는 것으로 알려져 있습니다.
딥러닝에서는 Embedding 이라는 방법을 이용하여 단어 관계 자체를 학습하는 방법을 사용합니다. 
이번 글에서는 머신러닝 방법에 좀 더 적합한 N-gram과 Bow에 대하여 알아보겠습니다.

+ n-gram : 문장에서 추출한 N개(또는 그 이하)의 연속된 단어 그룹. 
+ n-gram 예시 : 
  + "The cat sat on the mat." 문장으로 2-gram과 3-gram을 만드는 방법에 대하여 알아보겠습니다.
  + n-gram의 정의는 `N개 이하`의 연속된 단어 그룹
  + 2-gram : {"The", "The cat", "cat", "cat sat", "sat", "sat on", "on", "on the", "the", "the mat", "mat"}
  + 3-gram : {"The", "The cat", "cat", "cat sat", "The cat sat", "sat", "sat on", "on", "cat sat on", "on the", "the", "sat on the", "the mat", "mat", "on the mat"}
  + 여기서 다루는 것은 토큰의 집합이고 토큰에는 순서가 없습니다. 이런 종류의 토큰화 방법을 `BoW(Bag of Words)` 라고 합니다.
+ BoW : 순서가 없는 토큰화 방법
	+ 문장의 일반적인 구조가 사라짐
	+ 얕은 학습 방법의 언어 처리 모델에 사용이 적합함
		+ Logistic Regression, Random Forrest ...
