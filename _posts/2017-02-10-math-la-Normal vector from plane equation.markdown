---
layout: post
title: 평면 방정식의 법선 벡터
date: 2017-02-05 00:00:00
img: math/la/linear-algebra-image.jpg
categories: [math-la] 
tags: [Linear algebra, 선형대수학, 벡터, 내적, 외적, 삼중적] # add tag
---

<iframe src="//partners.coupang.com/cdn/redirect?url=customjs%2Faffiliate%2Fsearch-bar%2F0.0.3%2Flogo-01.html%3FtrackingCode%3DAF1042200" width="100%" height="85" frameborder="0" scrolling="no"></iframe>

<br><br>

출처 : 칸 아카데미 선형대수학 (https://ko.khanacademy.org/math/linear-algebra/vectors-and-spaces)

+ [선형대수학 전체 글 목록](https://gaussian37.github.io/math-la-Linear-Algebra-Table/) 

+ 이번 글에서는 **평면방정식의 법선벡터**에 대하여 알아보도록 하겠습니다.
+ 이번 글에서는 평면 방정식 $$ Ax + By + Cz = D $$ 가 있을 때, 이 평면에 Normal한 법선 벡터의 식을 유도합니다.
    + 법선 벡터 $$ \vec{n} = A\hat{i} + B\hat{j} + C\hat{k} $$ 입니다.

<img src="../assets/img/math/la/Normal vector form plane equation/1.PNG" alt="Drawing" style="width: 600px;"/>

+ 위 슬라이드와 같이 공간상에 노란색 평면이 있다고 가정하겠습니다.
+ 평면 상에 `노란색 점`과 `초록색 점`이 존재한다고 가정하겠습니다.
    + 노란색 점 $$ (x, y, z) $$과 초록색 점 $$ (x_{P}, y_{P}, z_{P}) $$을 원점과 연결하여 `벡터` 2개를 위 슬라이드 처럼 만듭니다.
    + 노란색 벡터 $$ \vec{P} = x\hat{i} + y\hat{j} + z\hat{k} $$
    + 초록색 벡터 $$ \vec{P_{1}} = x_{P}\hat{i} + y_{P}\hat{j} + z_{P}\hat{k} $$
+ 노란색 점과 초록색 점을 이은 벡터는 평면 상에 존재 하게 됩니다.
    + 하늘색 벡터 $$ \vec{P} - \vec{P_{1}} $$는 평면상에 존재합니다.
+ 평면과 수직인 보라색 법선 벡터 $$ \vec{n} = a\hat{i} + b\hat{j} + c\hat{k} $$ 라고 하겠습니다.

<br><br>

<img src="../assets/img/math/la/Normal vector form plane equation/2.PNG" alt="Drawing" style="width: 600px;"/>

+ 평면상의 하늘색 벡터와 법선 벡터는 직교하므로 **내적은 0**이 되어야 합니다.
+ 하늘색 벡터 $$ \vec{P} - \vec{P_{1}} = (x - x_{P})\hat{i} + (y - y_{P})\hat{j} + (z - z_{P})\hat{k} $$
+ ·$$ \vec{n} \cdot (\vec{P} - \vec{P_{1}}) = ax - ax_{P} + by - by_{P} + cz - cz_{P} = 0 $$
    + 하늘색 벡터와 보라색 법선 벡터를 내적합니다.
+ ·$$ ax + by + cz = ax_{P} + by_{P} + cz_{P} $$
    + 두 벡터의 내적한 결과의 좌변과 우변이 x, y, z 각각에 대응 되도록 정리합니다.
+ 평면의 방정식이 $$ Ax + By + Cz = D $$ 이므로 위 식과 대응해서 보면
    + a와 A, b와 B, c와 C 그리고 우변 전체($$ ax_{P} + by_{P} + cz_{P} $$)와 D를 대응할 수 있습니다.
+ 따라서 어떤 평면의 방정식이 주어진다면 그에 수직인 법선 벡터는 $$ \vec{n} = A\hat{i} + B\hat{j} + C\hat{k} $$ 임을 알 수 있습니다.
    + 즉 `계수만 뽑아내면 식을 정의할 수 있습니다.`
+  평면 방정식의 D는 **면을 이동시키기는 하지만 면의 기울기에는 영향을 미치지 않습니다.**
    + 따라서 법선 벡터를 정의하는 데 전혀 영향을 끼치지 않으므로 어떤 값이 있더라도 상관 없습니다.
+ 그래프 아래에 보면 간단한 예제가 있으니 참조하시면 됩니다.