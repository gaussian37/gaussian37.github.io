---
layout: post
title: opencv-python 코드 snippets
date: 2020-05-21 00:00:00
img: vision/opencv/opencv-python.png
categories: [vision-opencv] 
tags: [opencv, python, snippets] # add tag
---

<br>

[opencv 글 목록](https://gaussian37.github.io/vision-opencv-table/)

<br>

## **목차**

<br>

- ### window 창 크기 조절하는 방법
- ### contrast와 brightness 변경 방법

<br>

## **window 창 크기 조절하는 방법**

<br>

- 이미지의 height, width의 크기는 유지한 상태로 window에서 이미지를 크게 보고 싶으면 아래 코드를 통하여 창의 크기를 변경할 수 있습니다.
- 이 때, window의 크기만 조정하고 이미지의 크기는 변경하지 않으므로, 화면에 보이는 픽셀 한 개의 크기가 확대/축소 되어 전체 window의 크기가 변경된다고 이해하면 됩니다. 따라서 확대할 경우 한 픽셀의 크기가 커져서 보이는 것을 확인할 수 있습니다.

<br>

```python
cv2.namedWindow("image", 0)
cv2.resizeWindow("image", win_width, win_height)
```

<br>

- 위 코드와 같은 경우 window의 크기가 `win_width`, `win_height` 크기로 변경됩니다. 물론 이미지의 실제 크기 변경은 없습니다.

<br>

## **contrast와 brightness 변경 방법**

<br>

- 참조 : https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
- 픽셀의 값에 곱을 하면 `contrast`가 조절되고 덧셈을 하면 `brightness`가 조절됩니다.

<br>

$$ g(i,j) = \alpha \cdot f(i,j) + \beta $$

<br>

- 위 수식과 같이 각 픽셀에 대하여 $$ \alpha $$ 값을 곱해주면 `contrast`가 조정되고 $$ \beta $$ 값을 더해주면 `brightness`가 조정됩니다.
- 코드를 간단히 살펴보면 다음과 같습니다. (물론 아래 코드를 사용하진 않습니다. 느리기 때문입니다.)

<br>

```python
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        for c in range(image.shape[2]):
            new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
```

<br>

- 실제 `opencv`에서 제공하는 코드는 다음과 같습니다.

<br>

```python
new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
```

<br>

- 위 코드는 for-loop을 사용하지 않으면서도 opencv 내부적으로 잘 구현되어 있어서 for-loop의 naive한 버전 보다 상당히 빠르게 contrast와 brightness를 적용할 수 있습니다.

<br>

[opencv 글 목록](https://gaussian37.github.io/vision-opencv-table/)

<br>
