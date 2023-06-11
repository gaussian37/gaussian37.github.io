---
layout: post
title: opencv-python 코드 snippets
date: 2018-07-21 00:00:00
img: vision/opencv/opencv-python.png
categories: [vision-opencv] 
tags: [opencv, python, snippets] # add tag
---

<br>

[opencv 글 목록](https://gaussian37.github.io/vision-opencv-table/)

<br>

## **목차**

<br>

- ### [warpAffine을 이용한 기하학적 변환](#warpaffine을-이용한-기하학적-변환-1)
- ### [warpAffine과 warpPerspective를 이용한 기하학적 변환](#warpaffine과-warpperspective를-이용한-기하학적-변환-1)
- ### [resize를 이용한 크기 변환](#resize를-이용한-크기-변환-1)
- ### [flip을 이용한 대칭 변환](#flip을-이용한-대칭-변환-1)
- ### [window 창 크기 조절하는 방법](#window-창-크기-조절하는-방법-1)
- ### [contrast와 brightness 변경 방법](#contrast와-brightness-변경-방법-1)
- ### [이미지 붙이기(hconcat, vconcat)](#이미지-붙이기hconcat-vconcat-1)
- ### [OpenCV 한글 쓰기](#opencv-한글-쓰기-1)
- ### [90도 이미지 회전](#90도-이미지-회전-1)
- ### [gif 만들기 with imageio](#gif-만들기-with-imageio-1)
- ### [두 이미지를 오버레이 하기](#두-이미지를-오버레이-하기-1)
- ### [픽셀의 하한, 상한 값 정하기](#픽셀의-하한-상한-값-정하기-1)
- ### [Automatic Canny Edge Detection](#automatic-canny-edge-detection-1)
- ### [erosion과 dilation](#erosion과-dilation-1)
- ### [rvecs와 Rodrigues 함수](#rvecs와-rodrigues-함수-1)
- ### [기타 참조 내용](#기타-참조-내용-1)

<br>

## **warpAffine을 이용한 기하학적 변환**

<br>

- Affine 변환에 대한 이론적 개념은 아래 링크를 참조하시기 바랍니다.
    - 링크 : [https://gaussian37.github.io/vision-concept-geometric_transformation/](https://gaussian37.github.io/vision-concept-geometric_transformation/)

<br>

#### **Translation Transformation (이동 변환)**

<br>

- Affine 변환을 이용하여 `이동 변환`을 구현해 보도록 하겠습니다. 사용 방법은 다음과 같습니다.

<br>
<center><img src="../assets/img/vision/opencv/snippets/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- `src`는 입력 영상을 뜻하고 `M`은 이동 변환을 위한 affine 변환 행렬 입니다. 위의 이론과 관련된 링크를 참조하면 Affine 변환에서 사용되는 2 X 3 크기의 affine 변환 행렬의 의미를 확인하실 수 있습니다. affine 변환 행렬은 `np.float32`로 선언되어야 합니다.
- `dsize`는 출력 영상의 크기이며 `dst`는 출력 영상이 저장되는 array 입니다. 이 때, `dsize`가 `src`의 사이즈와 다르면 resize를 해야 하는데 그 때 사용되는 `interpolation` 방법이 `flags`입니다. 기본적으로 bilinear interpolation이 사용됩니다.
- 이동 변환을 하게 되는 경우 dst 사이즈 영역에서 값이 존재하는 영역이 있는 반면 값이 없는 영역도 발생하게 됩니다. 이 영역을 어떤 값으로 채울 지가 `borderValue`에 해당하며 기본값은 검정색인 0이 됩니다.

<br>

```python
src = cv2.imread('test.png')

'''
x' ← x + 200
y' ← y + 100

[1, 0, 200
 0, 1, 100]
'''
aff = np.array([[1, 0, 200], [0, 1, 100]], dtype=np.float32)
dst = cv2.warpAffine(src, aff, (0, 0))
```

<br>
<center><img src="../assets/img/vision/opencv/snippets/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 결과와 같이 $$ x $$ 방향으로 200, $$ y $$ 방향으로 100만큼 이동 변환하였고, 픽셀값이 없는 영역은 검은색 (0)으로 입력된 것을 확인할 수 있습니다.

<br>

#### **Shear Transformation (전단 변환)**

<br>

- 먼저 [전단 변환](https://gaussian37.github.io/vision-concept-geometric_transformation/)과 관련된 이론적인 내용은 링크를 참조하시기 바랍니다. 위 링크를 통하여 전단 변환 적용 시 어떻게 Affine 행렬을 작성해야 할 지 알 수 있습니다.
- 아래 예제는 x축을 y축 대비 0.5의 비율로 기울인 효과를 나타냅니다.

<br>

```python
src = cv2.imread('test.png') 

'''
x' ← x + 0.5 * y
y' ← y 

[1, 0.5, 0
 0, 1, 0]
'''
aff = np.array([[1, 0.5, 0], [0, 1, 0]], dtype=np.float32)
h, w = src.shape[:2] 

# dst의 크기는 affine 변환 행렬에서 x축 방향으로 늘어난 만큼 더 더해주어야 합니다.
# affine 변환 행렬에서 x축의 사이즈가 늘어난 크기는 y축 사이즈의 반 만큼 늘어나게 되므로 (h*0.5)를 w에 더해줍니다.
dst = cv2.warpAffine(src, aff, (w + int(h * 0.5), h))
```

<br>
<center><img src="../assets/img/vision/opencv/snippets/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 따라서 위 코드의 Affine 변환을 적용하였을 때, 위 그림과 같이 Shear Transformation이 적용됩니다.

<br>

#### **Rotation Transformation (회전 변환)**

<br>

- 먼저 회전 변환에 관한 이론적인 배경은 아래 링크를 참조하시기 바랍니다.
    - 링크 : [https://gaussian37.github.io/vision-concept-geometric_transformation/#rotation-transformation-회전-변환-1](https://gaussian37.github.io/vision-concept-geometric_transformation/#rotation-transformation-%ED%9A%8C%EC%A0%84-%EB%B3%80%ED%99%98-1)
- 앞에서 설명한 `warpAffine`을 이용하면 affine 변환 행렬만 회전 변환 행렬에 맞게 사용 하면 됩니다.
- 아래 코드와 같이 회전할 각도를 radian으로 정한 뒤 affine 변환 행렬을 만들어서 `warpAffine`에 적용하면 회전 변환을 적용할 수 있습니다.

<br>

```python
rad = 20 * math.pi / 180 
aff = np.array([[math.cos(rad), math.sin(rad), 0], 
                [-math.sin(rad), math.cos(rad), 0]], dtype=np.float32)
dst = cv2.warpAffine(src, aff, (0, 0))
```

<br>
<center><img src="../assets/img/vision/opencv/snippets/8.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 회전 결과는 직교 좌표계에서 (0, 0)을 기준 축으로 두고 회전을 하게 됩니다. 이와 같은 경우 회전되는 방향으로 많은 양의 이미지가 잘리게 됩니다.
- 이상적으로 이미지를 회전 하려면 **이미지의 중앙 좌표를 기준으로 회전**을 하는 것이 합리적으로 보입니다. 이미지의 임의의 점 (ex. 중앙 좌표)를 기준으로 회전 하려면 다음과 같은 과정 (① 이동 변환 ② 회전 변환 ③ 이동 변환)을 거쳐서 회전을 하게 됩니다.

<br>
<center><img src="../assets/img/vision/opencv/snippets/9.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이와 같은 변환을 하기 위해서 직접 affine 변환 행렬을 만들어도 상관 없고 `getRotationMatrix2d` 함수를 사용해도 됩니다.

<br>
<center><img src="../assets/img/vision/opencv/snippets/10.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 affine 변환 행렬에서 `(x, y)`는 이미지에서 **회전 축**을 의미합니다. 
- `angle`은 degree 단위의 각도로 **반시계 방향**으로 30도 회전이 필요하면 30을 입력합니다. 시계 방향은 음수 각도로 입력하면 됩니다.
- `scale`은 회전하면서 영상을 확대할 지, 축소할 지에 대한 scale 값입니다. 보통 영상을 회전하면 잘리는 영역이 발생하기 때문에 축소하여 회전된 영상을 확인하곤 합니다.

<br>

- 그러면 `getRotationMatrix2d`와 `warpAffine` 함수를 어떻게 섞어서 사용하는 지 살펴보겠습니다.

<br>

```python
center_point = (src.shape[1] / 2, src.shape[0] / 2)
# 20도 반시계 방향 회전 + scale 변환 없음
affine_matrix = cv2.getRotationMatrix2D(cp, 20, 1) 
dst = cv2.warpAffine(src, affine_matrix, (0, 0))
```

<br>
<center><img src="../assets/img/vision/opencv/snippets/11.png" alt="Drawing" style="width: 800px;"/></center>
<br>

## **warpAffine과 warpPerspective를 이용한 기하학적 변환**

<br>

- 먼저 Affine Transformation과 Perspective Transformation에 관한 개념적인 내용은 아래 링크를 통해 확인할 수 있습니다.
    - 링크 : [Affine Transformation과 Perspective Transformation](https://gaussian37.github.io/vision-concept-geometric_transformation/#affine-transformation%EA%B3%BC-perspective-transformation-1)
- 먼저 Affine Transformation을 하기 위하여 사용하는 2가지 함수는 `getAffineTransform` 함수와 `warpAffine` 함수입니다.

<br>

- `getAffineTransform` 함수는 원 영상의 3개의 좌표 $$ (x_{1}, y_{1}), (x_{2}, y_{2}), (x_{3}, y_{3}) $$와 변환된 영상에서 3개의 좌표에 대응되는 좌표 $$ (x_{1}', y_{1}'), (x_{2}', y_{2}'), (x_{3}', y_{3}') $$를 입력으로 주면 2 X 3 형태의 Affine 변환 행렬을 반환합니다.

<br>
<center><img src="../assets/img/vision/opencv/snippets/12.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- `getAffineTransform`을 통해 얻은 Affine 변환 행렬을`warpAffine`의 2번째 인자 `M`에 넣으면 Affine Transformation을 할 수 있습니다.

<br>
<center><img src="../assets/img/vision/opencv/snippets/13.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- Perspective Transformation을 할 때에도 방식은 유사합니다. `getPerspectiveTransform` 함수를 통하여 perspective 변환 행렬을 얻고 `warpPerspective`함수를 이용하여 변환을 합니다.
- `getPerspectiveTransform` 함수는 원 영상의 4개의 좌표 $$ (x_{1}, y_{1}), (x_{2}, y_{2}), (x_{3}, y_{3}), (x_{4}, y_{4}) $$와 변환된 영상에서 4개의 좌표에 대응되는 좌표 $$ (x_{1}', y_{1}'), (x_{2}', y_{2}'), (x_{3}', y_{3}'), (x_{4}', y_{4}') $$를 입력으로 주면 3 X 3 형태의 Perspective 변환 행렬을 반환합니다.

<br>
<center><img src="../assets/img/vision/opencv/snippets/14.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- `getAffineTgetPerspectiveTransformransform`을 통해 얻은 Affine 변환 행렬을`warpPerspective`의 2번째 인자 `M`에 넣으면 Perspective Transformation을 할 수 있습니다.

<br>
<center><img src="../assets/img/vision/opencv/snippets/15.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- Perspective Transformation을 OpenCV 코드로 어떻게 사용하는 지 예제를 살펴보겠습니다.

<br>

```python
src = cv2.imread('namecard.jpg')
w, h = 720, 400 

# 원 영상에서 4개의 점
srcQuad = np.array([[325, 307], [760, 369], [718, 611], [231, 515]], np.float32)
# perspective 변환 시 대응 되어야 할 점 (이미지의 각 모서리에 맞추는 예제)
dstQuad = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32) 
pers = cv2.getPerspectiveTransform(srcQuad, dstQuad) 
dst = cv2.warpPerspective(src, pers, (w, h))
```

<br>

## **resize를 이용한 크기 변환**

<br>

- 앞에서 다룬 Affine 변환 중 크기 변환은 기하학적 변환 중 가장 많이 사용되는 변환 중 하나입니다.
- 따라서 OpenCV에서는 크기 변환을 위한 별도 함수인 `resize`를 제공합니다.

<br>
<center><img src="../assets/img/vision/opencv/snippets/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 주의하실 점은 2번째 인자인 `dsize` 입니다. 먼저 출력 영상의 size를 `(w, h)` 형태로 명시적으로 입력할 수 있습니다. 하지만 만약 dsize = (0, 0)이 입력된다면 이후에 입력되는 `(fx, fy)`를 통하여 출력 영상의 확대/축소 비율을 정할 수 있습니다. 이 값은 $$ x $$와 $$ y $$ 방향의 scale factor 입니다. 따라서 실제 출력되는 크기인 dsize = (w, h)가 입력되거나 dsize = (0, 0) && 확대/축소 비율 (fx, fy)이 반드시 입력되어야 합니다.
- resize의 마지막 인자는 `interpolation` 입니다. 앞에서 다룬 affine 변환에서는 `flags` 인자에서 interpolation을 다룹니다. interpolation은 이미지의 기하학적 변환이 발생할 때 (특히, 영상의 크기가 커질 때), 중간 중간에 채워지지 않는 값들을 어떻게 채워 나아갈 지에 대한 방법입니다.
- `affine 변환`과 affine 변환 중 하나인 `resize` 모두 기본 interpolation 방법은 `bilinear interpolation(양선형 보간법)`입니다. bilinear interpolation은 interpolation 성능이 가장 좋은 방법은 아니지만 매우 효율적인 방법이면서 어느 정도 성능을 보장하기 때문에 많이 사용하는 방법입니다.
- OpenCV에서 제공하는 5가지 interpolation 방법은 위 테이블을 참조하시면 됩니다. 아래에 있는 방법일수록 효율성은 떨어지지만 interpolation 성능은 높아집니다. 마지막의 INTER_AREA는 영상 축소 시 효과적입니다.

<br>

```python
src = cv2.imread('rose.bmp') # 480x320 
dst1 = cv2.resize(src, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST) 
dst2 = cv2.resize(src, (1920, 1280)) # cv2.INTER_LINEAR 
dst3 = cv2.resize(src, (1920, 1280), interpolation=cv2.INTER_CUBIC) 
dst4 = cv2.resize(src, (1920, 1280), interpolation=cv2.INTER_LANCZOS4)
```

<br>
<center><img src="../assets/img/vision/opencv/snippets/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 각 interpolation 방식에 따른 resize 결과를 살펴보면 INTER_NEAREST은 artifact가 관찰되는 문제가 있는 반면 나머지 방식은 큰 품질 차이가 보이진 않습니다.

<br>
<center><img src="../assets/img/vision/opencv/snippets/6.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 앞에서 설명한 바와 같이 이미지를 축소할 때에는 `INTER_AREA` 방식의 interpolation을 사용하는 것이 좋습니다. 이 방법의 특성상 위 그림과 같이 어떤 형상이 1 ~ 2 픽셀로 이루어진 경우 영상 축소 시 디테일이 사라지는 문제를 개선할 수 있기 때문입니다.

<br>

## **flip을 이용한 대칭 변환**

<br>

- 영상의 대칭 변환을 사용하려면 `flip` 함수를 이용하면 쉽게 구현할 수 있습니다.
- `flip`은 기본적으로 `좌우 대칭`, `상하 대칭`, `좌우 + 상하 대칭`이 있습니다. 사용 방법은 다음과 같습니다.

<br>
<center><img src="../assets/img/vision/opencv/snippets/7.png" alt="Drawing" style="width: 800px;"/></center>
<br>


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

## **90도 이미지 회전**

<br>

- 파이썬에서 이미지를 90도 회전하는 방법은 opencv를 이용하는 방법과 numpy를 이용하는 방법이 있습니다.
- 이 글에서는 opencv를 이용하는 방법에 대하여 간략하게 설명하겠습니다.
- 사용하는 함수는 `cv2.rotate()` 함수이며 사용 방법은 `target_image = cv2.rotate(src_image, 옵션)`로 사용합니다.
- 사용 가능한 옵션은 다음과 같습니다.
    - `cv2.ROTATE_90_CLOCKWISE` : 90도 시계 방향 회전
    - `cv2.ROTATE_90_COUNTERCLOCKWISE` : 90도 반시계 방향 회전
    - `cv2.ROTATE_180` : 180도 회전
<br>

```python
import cv2

img = cv2.imread('data/src/lena.jpg')

img_rotate_90_clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
img_rotate_90_counterclockwise = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
img_rotate_180 = cv2.rotate(img, cv2.ROTATE_180)
```

<br>

## **gif 만들기 with imageio**

<br>

- 이미지를 이용하여 GIF를 만들 때에는 `imageio`라는 패키지를 이용하면 쉽게 만들 수 있습니다.
- 설치 : `pip install imageio`
- `imageio.mimsave()` 함수를 통하여 gif 형태로 저장할 수 있습니다. 기본적으로 파일 경로와 gif로 만들 이미지 리스트를 파라미터로 받습니다.
- 프레임 간 간격은 `duration`이란 옵션을 추가적으로 적용하여 간격 조정을 할 수 있습니다.

<br>

```python
import imageio
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('/path/to/movie.gif', images)
# imageio.mimsave('/path/to/movie.gif', images, duration=0.1)
```

<br>

## **두 이미지를 오버레이 하기**

<br>

- 두 이미지를 투명한 형태로 오버레이 하여 하나의 이미지에서 겹쳐서 보는 방법을 blending 이라고 합니다.
- 두 이미지를 blending 하기 위해서는 다음과 같은 간단한 연산을 통해 구현 가능합니다.

<br>

- $$ dst = \alpha * src1 + \beta * src2 + \gamma $$

- $$ \alpha + \beta = 1, \ \ 0 \le \alpha, \beta \le 1 $$

<br>

- 위 식을 살펴보면 각 영상에 $$ \alpha, \beta $$와 같은 가중치가 있습니다. 이 가중치가 1에 가까울수록 해당 영상을 좀 더 진하게 반영하고 0에 가까울수록 투명하게 반영됩니다.
- 위 식의 $$ \gamma $$는 bias에 해당합니다.
- blending을 하기 위해서 `cv2.addWeighted`를 사용하며 사용 방법은 다음과 같습니다.
- `dst = cv2.addWeighted(src1, alpha, src2, beta, gamma)`
- 위 함수는 src1과 src2 영상을 각각 $$ \alpha, \beta $$의 가중치를 사용하여 blending 합니다. 

<br>

## **픽셀의 하한, 상한 값 정하기**

<br>

- 이미지 데이터에 어떤 연산을 가했을 때, uint8 데이터 타입의 값 범위인 [0, 255]를 벗어날 수 있습니다.
- 범위에 벗어난 값을 하한값, 상한값으로 saturation 시킬 때, `np.clip(array, lower_bound, upper_bound)`을 이용할 수 있습니다.

<br>

```python
lower_bound = 0
upper_bound = 255
img = np.clip(img, lower_bound, upper_bound).astype(np.uint8)
```

<br>

## **Automatic Canny Edge Detection**

<br>

- 아래는 `canny edge detection`을 위한 코드 입니다. `canny`에서 필요한 하한값은 $$ -\sigma $$ 을 사용하고 상한값 $$ \sigma $$ 를 사용하여 각각의 이미지의 통계에 맞게 사용할 수 있도록 적용합니다.

<br>

```python
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged
```

<br>

## **erosion과 dilation**

<br>

- 아래는 `erosion`과 `dilation`을 위한 간단한 코드 입니다.

<br>

```python
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
dilation = cv2.dilate(img, kernel, iterations=2)
```

<br>

## **rvecs와 Rodrigues 함수**

<br>

- 참조 : https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=rodrigues#void%20Rodrigues(InputArray%20src,%20OutputArray%20dst,%20OutputArray%20jacobian)
- 참조 : https://www.andre-gaschler.com/rotationconverter/

- `opencv`에서 캘리브레이션이나 3D 관련 함수 사용 시 `rvecs`와 `tvecs`를 입/출력 값으로 사용하곤 합니다.
- `rvecs`와 `tvecs`는 `opencv`함수에서 계산된 결과 또는 계산하고자 하는 $$ X, Y, Z $$ 축 순서의 `Rotation`과 `Translation`을 위한 정보입니다.

<br>

- `rvecs`와 `tvecs`의 타입은 `list`이며 각 원소는 3개의 값을 가지는 벡터입니다. `list`는 처리한 이미지의 갯수를 의미합니다. 예를 들어 어떤 작업 (ex. 캘리브레이션)을 10장의 이미지로 처리하였으면 `list`의 길이는 10이 되며 이미지 각 정보에 대한 `Rotation`, `Translation`이 구성되어 입/출력됩니다.
- 만약 $$ i $$ 번째 이미지의 `Rotation`과 `Translation`을 구성한다고 하면 `rvces[i]`와 `tvecs[i]`가 대응이 되어 $$ X, Y, Z $$ 의 `Rotation`과 `Translation`을 구성할 수 있습니다. 관련 방법을 좀 더 자세하게 살펴보겠습니다.

<br>

- `rvecs[i] = rvec`는 3개의 값을 가지는 벡터입니다. 이 벡터 값은 `Rodrigues` 변환을 통해 $$ 3 \times 3 $$ 크기의 행렬로 만들 수 있습니다.
- 먼저 `rvec`은 `로드리게스 회전 (또는 축각 회전)`으로 표현되어 있습니다. 이 방법은 3차원에서 회전하고자 하는 3차원 벡터 축(Axis)과 축을 기준으로 회전하고자 하는 회전량을 이용하는 방법입니다. 상세 내용은 아래 글을 참조하시기 바랍니다.
    - [로드리게스 회전 (축각 회전) 설명](https://gaussian37.github.io/vision-concept-axis_angle_rotation/)
- 따라서 `로드리게스 회전`을 위해서는 기본적으로는 회전축 `axis`를 표현하는 벡터값 3개와 회전각도 1개 총 4개의 값이 필요로 합니다. 하지만 `rvec`은 3개의 값만 가지고 있는데, 이 부분은 `opencv`에서 정보를 compact한 방식으로 압축하여 사용하기 때문이며 다음과 같이 4개의 값을 만들면 되는 것으로 소개됩니다.

<br>

- $$ \text{rvec} = (a, b, c) $$

- $$ \text{Rotation Angle : } \theta = \sqrt{a^{2} + b^{2} + c^{2}} $$

- $$ \text{Rotation Axis : } v = (\frac{a}{\theta}, \frac{b}{\theta}, \frac{c}{\theta}) $$

<br>

- 그러면 각각 3개의 원소를 가지는 `rvec`과 `tvec`을 이용하여 어떻게 $$ R \vert t $$ 행렬을 만드는 지 살펴보도록 하겠습니다.
- 먼저 실제 사용하는 `OpenCV` 함수를 사용하여 로드리게스 회전을 구하려면 `cv2.Rodrigues`를 사용하면 됩니다.

<br>

```python
rvec
# array([[-0.18105588],
#        [-0.12723858],
#        [-1.53333626]])

tvec
# array([[-34.65648128],
#        [  4.21510712],
#        [215.64322146]])

R, _ = cv2.Rodrigues(rvec)
# [[ 0.03493564  0.99890668  0.0310637 ]
#  [-0.98012378  0.02817188  0.1963765 ]
#  [ 0.19528667 -0.03730682  0.98003639]]

Rt = np.identity(4)
Rt[:3, :3] = R
Rt[:3, 3] = tvec.reshape(-1)
# [[  0.03493564   0.99890668   0.0310637  -34.65648128]
#  [ -0.98012378   0.02817188   0.1963765    4.21510712]
#  [  0.19528667  -0.03730682   0.98003639 215.64322146]
#  [  0.           0.           0.           1.        ]]
```

<br>

- 위의 `cv2.Rodrigues()` 함수는 `opencv`에서 정의한 `rvec` 형식을 따릅니다. 그러면 실제 `opencv`의 값과 계산기를 통해 구한 값이 같은 지 살펴보도록 하겠습니다. 계산기는 [3D Rotation Converter](https://www.andre-gaschler.com/rotationconverter/)를 사용하였습니다.

<br>

```python
rvec
# array([[-0.18105588],
#        [-0.12723858],
#        [-1.53333626]])

theta = np.linalg.norm(rvec)
# 1.5492227001175765

v = rvec / theta
# [[-0.11686885]
#  [-0.08213059]
#  [-0.98974554]]
```

<br>
<center><img src="../assets/img/vision/opencv/snippets/16.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 이미지의 빨간색 박스 위주로 보면 앞에서 구한 `Axis`와 `Angle`을 통해 `Rotation Matrix`를 구한 것이 `cv2.Rodrigues` 함수와 같은 결과임을 알 수 있습니다.

<br>

## **기타 참조 내용**

<br>

- python에서 opencv를 사용 중에 `cannot import name '_registerMatType' from 'cv2.cv2'`와 같은 에러가 발생하는 경우 `opencv-python-headless`를 설치하면 됩니다. head가 생겨서 2중으로 구성되어 (ex. cv2.cv2) 발생한 오류입니다. 아래 버전으로 install 하시면 문제를 해결할 수 있습니다.
    - 설치 : `pip install opencv-python-headless==4.1.2.30`

<br>

[opencv 글 목록](https://gaussian37.github.io/vision-opencv-table/)

<br>
