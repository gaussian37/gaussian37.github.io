---
layout: post
title: Optical Flow 알아보기
date: 2019-12-26 00:00:00
img: vision/concept/optical_flow/0.png
categories: [vision-concept] 
tags: [vision, optical flow] # add tag
---

<br>

- 참조 : 컴퓨터 비전 (오일석 저)
- 참조 : 영상신호처리특론 강의 (연세대 )

- 이번 글에서는 object의 움직임과 관련된 개념 중 하나인 `optical flow`에 대하여 알아보도록 하겠습니다.

<br>

## **목차**

<br>

- ### 연속 영상에 관한 지식
- ### Optical flow의 의미와 추정 원리
- ### Lucas-Kanade 알고리즘
- ### Horn-Schunck 알고리즘
- ### Optical flow의 확용
- ### Optical flow 성능 평가
- ### 딥러닝에서의 optical flow

<br>

## **연속 영상에 관한 지식**

<br>

- 먼저 연속된 영상이 입력될 때 사용되는 용어로 `Frame`이 있습니다. Frame은 연속 영상 (동영상)을 구성하는 각각의 영상을 뜻하며 시간 축 $$ t $$를 가지게 됩니다.

<br>
<center><img src="../assets/img/vision/concept/optical_flow/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림과 같이 기존에 이미지 좌표계인 $$ y, x $$ 이외의 $$ t $$ 축이 더 추가되며 1 초당 몇 번의 Frame을 다루는 지에 따라서 `FPS(Frame Per Second)`라는 단위를 사용합니다. 예를 들어 30 fps는 30 frame / sec 를 뜻합니다.
- 동영상에서 다루는 fps는 어떤 영상을 다루는 지에 따라서 다르게 설정됩니다. 예를 들어 일반 웹캡이나 감시 카메라 등은 일반적으로 10 ~ 30 fps를 사용합니다. 

<br>

- 연속된 Frame이 들어올 떄, `영상 일관성(coherence)`를 가집니다.

<br>
<center><img src="../assets/img/vision/concept/optical_flow/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 예를 들어 어떤 픽셀 (y, x)의 색이 빨간색이라면, 주변 픽셀의도 유사한 색을 띌 가능성이 높다는 것입니다.

<br>

- 연속된 Frame에서의 또 다른 성질로 `시간 일관성`을 가집니다.

<br>
<center><img src="../assets/img/vision/concept/optical_flow/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 즉, $$ t $$ 순간의 픽셀 값 $$ f(y, x, t) $$는 다음 순간 $$ f(y, x, t + 1) $$과 비슷할 가능성이 높습니다. 만약 픽셀값의 변경량이 거의 없다면 그 만큼 이동량이 작다고 직관적으로 생각할 수 있습니다.

<br>

- 보통 가장 직관적인 방법으로 두 Frame 간의 픽셀값의 차이를 통하여 변화를 확인하는 방법이 있습니다.
- 예를 들어 $$ r $$번째 Frame과 $$ t $$ 번째 Frame의 픽셀 값의 차이를 계산한다고 가정해 보겠습니다.

<br>

$$ f(n) = \begin{cases} 1, & \vert f(y, x, t) - f(y, x, r) \vert > \tau \\[2ex] 0, & \text{else} \end{cases} $$

<br>

<br>
<center><img src="../assets/img/vision/concept/optical_flow/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 식을 이용하여 영상 간의 차이를 구하면 영상에서의 움직임을 구할 수 있습니다. 이 경우는 굉장히 간단하지만 한계점도 명확히 있습니다. 
- 먼저, **위치가 고정된 카메라**에서만 적용 가능하고 **배경과 물체의 색상이나 명암에 큰 변화가 있는 경우 사용할 수 없습니다.**
- 따라서 두 영상의 차이를 구하는 한계를 개선하기 위하여 실제 영상에서 움직이는 물체를 찾을 필요가 있습니다.

<br>
<center><img src="../assets/img/vision/concept/optical_flow/5.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 물체의 이동이 발생할 때, 3차원 실제 공간에서 일어나는 물체의 움직임은 2차원 영상 공간에 투영됩니다. 이 때, 차원이 축소되면서 투영되기 때문에 3차원 공간의 무수히 많은 벡터가 2차원 영상 공간 상의 벡터로 투영될 수 있습니다.

<br>

- 여기서 `motion field`라는 개념이 필요합니다.

<br>
<center><img src="../assets/img/vision/concept/optical_flow/6.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- `motion field`는 움직임이 발생한 모든 점의 모션 벡터로 얻어낸 2차원 모션맵을 뜻합니다.
- 위 그림을 보면 초록색 삼각형( $$ t $$ )이 파란색 삼각형($$ t+1 $$)과 같이 이동하였을 떄, 픽셀들이 일관성 있게 이동한 것을 확인할 수 있습니다. 따라서 위 그림과 같은 상황에서는 이상적으로 motion field를 구할 수 있습니다. 물론 근본적으로 **현실 이미지에서는 위 그림과 같은 motion field를 구하기는 어렵습니다.** 대표적으로 `구체 회전`과 `광원 회전`과 같은 어려움이 있습니다.

<br>
<center><img src="../assets/img/vision/concept/optical_flow/7.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 구의 texture가 단조로운 상태에서 회전이 발생한다면 모션 벡터가 발생함에도 불구하고 영상에는 아무런 변화가 발생하지 않습니다.
- 반면 구가 회전하지 않지만 광원이 움직이는 경우에는 모션 벡터가 0이어야 하지만 영상에는 변화가 발생합니다.
- 즉, 실질적으로 움직였지만 모션 벡터가 발생하지 않는 경우와 움직임이 발생하지 않았지만 모션 벡터가 발생하는 오인식 경우가 문제가 되곤 합니다.

<br>

- 따라서 `Temporal Feature`를 구하는 것이 중요합니다. 이는 **시간의 흐름에 따라 변하는 특징**을 나타내며 움직이는 물체의 경우 잘 나타납니다.

<br>
<center><img src="../assets/img/vision/concept/optical_flow/8.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 앞에서 설명하였듯이, 픽셀의 움직임을 `motion`이라고 합니다. motion을 나타낼 떄에는 대표적으로 ① motion의 크기, ② X방향, Y방향의 모션, ③ Dominant Motion등을 나타냅니다.

<br>

## **Optical flow란 무엇인가**

<br>

- 앞에서 설명한 `Motion field`를 `Optical field`라고도 합니다. Optical Field는 수식으로 나타낼 수 있고 이 내용은 글의 뒷부분에서 다룰 예정입니다.
- Optical flow는 Optical field를 구하기 위하여 이전 프레임과 현재 프레임의 차이를 이용하고 픽셀값과 주변 픽셀들과의 관계를 통해 각 픽셀의 이동(motion)을 계산하여 추출합니다. 이를 통하여 움직임을 구별해 낼 수 있습니다.

<br>
<center><img src="../assets/img/vision/concept/optical_flow/9.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- Optical flow를 사용하면 위 그림 예시와 같이 차량의 움직임을 나타낼 수 있습니다. 그 결과 Optical flow가 좌측으로 발생하게 되는 것을 확인할 수 있습니다.


