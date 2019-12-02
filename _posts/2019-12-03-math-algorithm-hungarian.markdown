---
layout: post
title: 헝가리안(hungarian) 알고리즘, 최소 비용 작업 배분
date: 2019-12-03 00:00:00
img: math/algorithm/algorithm.png
categories: [math-algorithm] 
tags: [algorithm, 알고리즘, 헝가리안, hungarian] # add tag
---

<br>

- 출처 
- https://www.topcoder.com/community/competitive-programming/tutorials/assignment-problem-and-hungarian-algorithm/
- http://www.cse.ust.hk/~golin/COMP572/Notes/Matching.pdf
- http://mip.hnu.kr/courses/network/chap8/chap8.html
- https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
- https://skkuassa.tistory.com/186
- https://gazelle-and-cs.tistory.com/29

<br>

- 이번 글에서는 최소 비용으로 작업을 배분하는 문제인 헝가리안 알고리즘을 다루어 보도록 하겠습니다.
- 헝가리안 알고리즘에는 $$ O(N^{4}) $$의 복잡도를 가지는 방법과 $$ O(N^{3}) $$의 복잡도를 가지는 방법이 있습니다.
- 물론 worst case를 따져서 그렇지만 평균적으로는 $$ O(N^{4}) $$의 방법도 $$ N^{4} $$의 worst case 까지 빠지는 경우는 잘 없습니다.
- 각종 자료들을 찾아본 결과 대부분의 글에서 다루는 방법은 $$ O(N^{4}) $$의 방법이고 이 방법이 좀 더 직관적입니다.
- 그럴면 먼저 $$ O(N^{4}) $$의 방법을 먼저 다루어 보고 그 다음에 $$ O(N^{3}) $$의 방법을 다루어 보도록 하겠습니다.

<br>

## **목차**

<br>

- ### 문제 정의
- ### python 라이브러리를 통한 빠른 해결법
- ### $$ O(N^{4}) $$ 알고리즘 및 코드
- ### $$ O(N^{3}) $$ 알고리즘 및 코드