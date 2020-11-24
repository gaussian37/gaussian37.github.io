---
layout: post
title: Attention 모델과 Seq2seq with Attention
date: 2020-11-20 00:00:00
img: dl/concept/attention/0.png
categories: [dl-concept]
tags: [attention, seq2seq] # add tag
---

<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>

- 참조 : fastcampus 딥러닝/인공지능 올인원 패키지 Online.
- 참조 : 시퀀스 투 시퀀스 + 어텐션 모델 (https://www.youtube.com/watch?v=WsQLdu2JMgI&feature=youtu.be)
- 참조 : 모두를 위한 기계번역 (https://youtu.be/N4E53ZcUBJs)
- 참조 : Pytorch Seq2Seq Tutorial for Machine Translation (https://www.youtube.com/watch?v=EoGUlvhRYpk&feature=youtu.be)
- 참조 : Pytorch Seq2Seq with Attention for Machine Translation (https://youtu.be/sQUqQddQtB4)

<br>

- 이번 글에서는 `seq2seq`(Sequence 2 Sequence)에 어떻게 `Attention` 모델이 사용되는 지를 통하여 Attention의 메커니즘에 대하여 다루어 보겠습니다.

<br>
<center><img src="../assets/img/dl/concept/attention/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 번역 문제를 다룰 때, 기본적으로 사용할 수 있는 seq2seq 모델은 위 그림과 같습니다. 먼저 입력으로 word2vec과 같은 Word Embedding방법을 통하여 얻은 embedding을 Encoder에서 입력으로 받습니다. 각 단어에 해당하는 embedding은 벡터 형태로 dense representation을 가집니다.
- Encoder에서 각 단어의 embedding과 RNN의 hidden state를 거쳐서 정보가 압축이 되고 Encoder의 마지막 부분의 출력이 **context**가 됩니다.
- Decoder에서는 context를 이용하여 






<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>