---
layout: post
title: NeRF, Representing Scenes as Neural Radiance Fields for View Synthesis
date: 2022-09-02 00:00:00
img: vision/fusion/nerf/0.png
categories: [vision-fusion] 
tags: [nerf, neural radiance, 너프] # add tag
---

<br>

[fusion 관련 글 목차](https://gaussian37.github.io/vision-fusion-table/)

<br>

- 논문 : https://arxiv.org/abs/2003.08934
- 참조 : https://youtu.be/WSfEfZ0ilw4
- 참조 : https://www.youtube.com/watch?v=JuH79E8rdKc
- 참조 : https://www.matthewtancik.com/nerf
- 참조 : https://www.youtube.com/watch?v=zkeh7Tt9tYQ

<br>

## **목차**

<br>

- ### [NERF 개념 소개](#nerf-개념-소개-1)
- ### [NERF 논문 분석](#nerf-논문-분석-1)
- ### [NERF 코드 분석](#nerf-코드-분석-1)

<br>

## **NERF 개념 소개**

<br>

- `NERF`는 **Neural Radiance Fields for View Synthesis**을 의미합니다. `View Synthesis`에서 의미하는 바와 같이 새로은 View를 생성하게 되며 2D 이미지 여러장을 이용하여 3D 뷰를 생성해 내는 것을 의미합니다.

<br>
<center><img src="../assets/img/vision/fusion/nerf/1.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 영상과 같이 `NERF`를 이용하게 되면 2D 이미지를 위 영상과 같은 깊이 정보를 가진 3D 정보 형태로 형태로 나타낼 수 있으며 기존에 가지고 있는 2D 이미지들 사이 사이의 불연속적이어서 표현할 수 없는 부분들을 매끄럽게 만들어 갈 수 있습니다. 이러한 태스크를 `NVS(Novel View Synthesis)` 라고 부릅니다.

<br>
<center><img src="../assets/img/vision/fusion/nerf/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림과 같이 NERF는 입력에는 없는 `새로운 뷰`의 이미지를 생성해 낼 수 있습니다. (물론 새로운 뷰를 생성해 내는 것이지 완전히 새로운 것을 생성해 내지는 못합니다.) 같은 공간을 카메라의 다른 각도에서 보았을 때를 가정하여 `NVS` 하는 것으로 생각하면 됩니다. 따라서 위 그림은 카메라가 움직이는 것이라고 생각하시면 됩니다.

<br>
<center><img src="../assets/img/vision/fusion/nerf/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 따라서 `NERF`를 구현하기 위하여 다양한 카메라 뷰에서의 입력 이미지를 이용하여 적절할 학습 방법을 통해 입력 이미지에 없는 카메라 뷰에서의 상황을 재현해 보는 것이 이번 논문의 목표라고 할 수 있습니다.

<br>
<center><img src="../assets/img/vision/fusion/nerf/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 물론 `NERF` 이전에도 `view synthesis`를 위한 연구는 진행되었습니다. 하지만 360도 전체를 모두 렌덩할 때, 정면 이외의 부분에서는 부정확하게 렌더링되는 문제와 모든 부분을 저장하려고 할 때, 한 개의 상황에 대하여 너무 많은 저장 공간이 필요로 한다는 큰 단점이 있었습니다. 즉, 가능한 많은 모든 영역의 이미지가 필요로 하다는 뜻입니다. 
- 이러한 문제로 학습 방식으로 `view synthesis`를 할 때, 많은 이미지를 다 GPU를 통해 사용해야 한다는 단점도 존재하였습니다.

<br>

- `NERF`는 이러한 문제를 개선하기 위하여 explicit feature(representation) 대신에 `implicit feature(representation)`를 사용합니다.

<br>
<center><img src="../assets/img/vision/fusion/nerf/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- explicit feature는 3D 공간의 정보를 Voxel Representation을 통해 나타내는 반면에 `implicit feature`는 뉴럴 네트워크를 통해 간접적으로 저장하는 방식을 뜻합니다. 위 그림과 같이 `view synthesis`를 해야 할 카메라 뷰 기준에서의 `X, Y, Z`를 네트워크의 쿼리로 입력하고 그 위치에서의 컬러값을 출력으로 뽑는 것이 방법이 될 수 있습니다.
- 이와 같은 방법을 통하면 실제 필요한 것은 딥러닝 네트워크이고 무거운 Voxel Representation을 모두 저장할 필요는 없어집니다. 따라서 `implicit feature`를 위한 네트워크가 Voxel representation의 압축 기술이라고 말할 수도 있습니다. 추가적으로 Voxel Representation을 모두 저장할 때 발생하는 용량 문제를 해소하기 위해 해상도를 줄일 수 밖에 없던 문제 또한 `implicit feature`를 이용하면 개선할 수 있습니다.
- 또한 쿼리에 해당하는 X, Y, Z 값을 (공간 상에서) 연속적인 값으로 입력할 수 있기 때문에 출력 또한 이산적이지 않고 연속적으로 생성할 수 있어 입력에 사용되지 않은 뷰 또한 출력으로 만들 수 있습니다.
- 정리하면 `implicit feature`를 사용하였을 때, ① voxel representatio 보다 가벼우며 ② 공간 상에서 연속적인 출력이 가능하다는 장점이 있습니다.

<br>
<center><img src="../assets/img/vision/fusion/nerf/6.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- `NERF`는 `implicit feature`를 이용하는 논문들 중 카메라의 `position`과 `direction`에 따른 `R, G, B`, `volume density`를 출력하는 방식을 처음으로 사용하여 주목을 받았습니다. 
- 위 그림과 같이 5D Input인 $$ X, Y, Z $$ `Position`과 $$ \theta, \phi $$ `Direction`을 입력으로 받아서 Output인 $$ R, G, B $$ `Color`와 $$ \sigma $$ 인 `Density`를 출력합니다.
- 이 중간 과정에 사용된 뉴럴 네트워크는 $$ F_{\theta} $$ 로 본 논문에서는 `Fully Connected Layer` 즉, `MLP`만 사용하여 구현하였기 때문에 $$ F_{\theta} $$ 로 단순하게 나타냅니다.

<br>

[fusion 관련 글 목차](https://gaussian37.github.io/vision-fusion-table/)

<br>