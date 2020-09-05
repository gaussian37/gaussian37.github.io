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
- ### 이미지 붙이기(hconcat, vconcat)
- ### OpenCV 한글 쓰기

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

- $$ g(i,j) = \alpha \cdot f(i,j) + \beta $$

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

## **이미지 붙이기(hconcat, vconcat)**

<br>

- 이미지를 가로로 또는 세로로 붙이고 싶을 때, 쉽게 사용할 수 있는 함수가 `hconcat`과 `vconcat`이 있습니다.
- `hconcat`은 horizontal concatenate로 가로로 이미지를 붙이는 것이고 `vconcat`은 vertical concatenate로 세로로 이미지를 붙이는 것입니다.
- `hconcat`을 이용하려면 가로로 붙여야 하기 때문에 height가 같아야 합니다. 반면 `vconcat`은 세로로 이미지를 붙여야 하기 때문에 width가 같아야 합니다.

<br>

```python
image = cv2.hconcat([image1, image2, image3])
image = cv2.vconcat([image1, image2, image3])
```

<br>

## **OpenCV 한글 쓰기**

<br>

- OpenCV를 이용하여 이미지에 글자를 쓸 때, 한글은 안써집니다.
- 따라서 OpenCV 함수를 직접 이용하지 않고 PIL(Python Image Library)를 이용하여 우회해서 사용하면 사용 가능합니다.
- 아래 함수는 numpy와 pil을 이용하여 numpy 배열에 한글을 입력하는 함수 `PutText`를 작성한 것입니다.
- PutText 함수의 `fontpath` 의 font를 수정하면 원하는 font도 사용가능합니다.

```python
import numpy as np
from PIL import ImageFont, ImageDraw, Image
def PutText(src, text, font_size, xy, bgr):
    fontpath = "fonts/gulim.ttc"
    font = ImageFont.truetype(fontpath, font_size)
    src_pil = Image.fromarray(src)
    draw = ImageDraw.Draw(src_pil)
    draw.text(xy, text, font=font, fill=bgr)
    target = np.array(src_pil)
    return target

img = np.zeros((300,500,3),np.uint8)
img= PutText(img, "테스트입니다.", font_size = 30, xy = (100, 200), bgr = (255, 0, 0))
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()
```

<br>

[opencv 글 목록](https://gaussian37.github.io/vision-opencv-table/)

<br>
