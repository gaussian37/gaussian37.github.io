---
layout: post
title: Universal-Sentence-Encoder-Large 를 이용한 자연어 문장 처리
date: 2019-01-22 18:43:00
img: nlp/concept/Universal-Sentence-Encoder/Universal-Sentence-Encoder.png
categories: [python-django] 
tags: [nlp, 자연어 처리, tensorflow, tensorflow hub, universal sentence encoder, transformer] # add tag
---

# universal-sentence-encoder-large 란

+ Universal Sentence Encoder는 텍스트 분류, 의미론적 유사성, 클러스터링 및 기타 자연어 처리에 사용할 수있는 `고차원 벡터`로 **텍스트를 인코딩**합니다.
+ 이 모델은 문장, 구 또는 짧은 단락과 같이 단어 길이가 더 긴 텍스트에 대해 학습되고 최적화되었습니다. 
+ 입력은 가변 길이의 영어 텍스트이고 출력은 512 차원 벡터입니다. 
+ universal-sentence-encoder-large 모델은 Transformer 인코더로 학습되었습니다.
+ `Universal Sentence Encoder`는 단순히 단어가 아닌 시퀀스의 의미를 모델링 하였습니다.
+ 모듈의 크기는 약 800MB 이고 안번 로드하면 다음 부터 사용할 때에는 빠르게 사용할 수 있습니다.
