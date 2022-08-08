---
layout: post
title: SSIM (Structural Similarity Index)
date: 2022-02-17 00:00:00
img: vision/concept/ssim/0.png
categories: [vision-concept] 
tags: [SSIM, Structural Similarity Index] # add tag
---

<br>

- 참조 : https://en.wikipedia.org/wiki/Structural_similarity
- 참조 : https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e
- 참조 : https://bskyvision.com/878
- 참조 : https://walkaroundthedevelop.tistory.com/m/56
- 참조 : https://nate9389.tistory.com/2067
- 참조 : https://medium.com/@sanari85/image-reconstruction-%EC%97%90%EC%84%9C-ssim-index%EC%9D%98-%EC%9E%AC%EC%A1%B0%EB%AA%85-b3ca26434fb1

<br>

- 이번 글에서는 두 이미지를 비교하는 지표인 `SSIM`에 대하여 다루어 보도록 하겠습니다.
- `SSIM`은 `Structural Similarity Index`의 약어로 사용되며 주어진 2개의 이미지의 `similarity(유사도)`를 계산하는 측도로 사용됩니다.
- `SSIM`은 두 이미지의 단순 유사도를 측정하는데 사용하기도 하지만 풀고자 하는 문제가 두 이미지가 유사해지도록 만들어야 되는 문제일 때 `SSIM`을 Loss Function 형태로 사용하기도 합니다. 왜냐하면 `SSIM`이 gradient-based로 구현되어 있기 때문입니다.
- 딥러닝에서 **두 이미지를 유사하게 만드는 문제**나 depth estimation시 **disparity를 구하기 위하여 이미지를 복원**할 때, **두 이미지 또는 두 패치의 유사도를 측정**하여 Loss Function을 사용하는 방법이 많이 사용됩니다. 
- 따라서 이번 챕터에서는 `SSIM`의 원리에 대하여 먼저 알아보고 Pytorch의 구현 방법을 통하여 학습에 사용하는 방법과 skimage를 이용하여 단순히 이미지의 유사도를 측정하는 방법에 대하여 살펴보도록 하겠습니다.

<br>

## **목차**

<br>

- ### [SSIM의 정의](#ssim의-정의)
- ### [SSIM의 Pytorch에서의 사용법 (globally)](#ssim의-pytorch에서의-사용법-1)
- ### [SSIM의 Pytorch에서의 사용법 (locally)](#ssim의-pytorch에서의-사용법-1)
- ### [SSIM의 skimage에서의 사용법](#ssim의-skimage에서의-사용법-1)

<br>

## **SSIM의 정의**

<br>

- `SSIM`은 Structureal Similarity Index Measure의 약어로 두 이미지의 유사도를 `luminance`, `contrast`, `structure` 3가지 요소를 이용하여 비교하는 방법을 의미합니다. 이와 같은 요소를 이용하여 이미지를 비교하는 이유는 실제 인간의 시각 기관도 이와 같은 방법으로 인식하기 때문입니다.
- `SSIM`의 최종 결과는 0 ~ 1 사이이며 1에 가까울수록 두 이미지가 유사함을 의미합니다. 그러면 `luminance`, `contrast`, `structure`가 각각 어떻게 계산되어서 하나로 합쳐지는 지 살펴보도록 하곘습니다. 입력값의 범위에 따라서 -1 ~ 1 사이의 값을 가질수도 있으며 1에 가까울수록 두 이미지가 유사한 것은 동일합니다.

<br>

#### **Luminance**

<br>

- `luminance`는 한글로 휘도라고 하며 빛의 밝기를 나타내는 양입니다. SSIM에서 계산할 때, 별도 빛의 밝기 성분을 추출해서 사용하지는 않고 이미지의 픽셀값을 이용합니다. (픽셀 값이 클수록 밝음을 이용함) grayscale 이미지에서는 각 픽셀의 값을 의미하며 RGB 이미지에서는 R, G, B 각 채널 별 픽셀 값을 의미합니다.

<br>

- $$ \mu_{x} = \frac{1}{N}\sum_{i=1}^{N} x_{i} $$

- $$ x_{i} $$ : 각 픽셀의 값 (밝기 값을 의미함)

- $$ N $$ : 전체 픽셀의 갯수

- $$ \mu_{x} $$ : 이미지의 평균 `luminance`

<br>

- 두 이미지의 `luminance`가 얼마나 다른 지 비교하기 위해 $$ \mu_{x} $$ 값을 이용합니다. 두 이미지를 $$ x, y $$ 라고 할 때 두 이미지의 `luminance`를 비교하기 위한 식은 다음과 같습니다.

<br>

- $$ l(x, y) = \frac{2\mu_{x}\mu_{y} + C_{1}}{\mu_{x}^{2}, \mu_{y}^{2} + C_{1}} \tag{1} $$

<br>

- 식 (1)의 $$ C_{1} $$ 을 제외하고 살펴보면 $$ \frac{2\mu_{x}\mu_{y}}{\mu_{x}^{2}, \mu_{y}^{2}} $$ 가 되며 $$ \mu_{x}, \mu_{y} $$ 각 같으면 1이 되고 두 값이 차이가 많이 날수록 0에 가까워 집니다. 이와 같은 성질을 이용하여 두 값의 차이에 따른 값의 범위를 0 ~ 1로 계산될 수 있도록 합니다.
- 식 (1)에서 $$ C_{1} $$ 의 사용 용도는 분모에 0이 되는 것을 방지하기 위하여 안정성을 위해 추가하였고 값의 정의 방법은 아래 식을 따르는 것으로 알려져 있습니다. (하지만 크게 중요하지 않으니 적당한 상수값을 사용하여도 무관합니다.)

<br>

- $$ C_{1} = (K_{1}L)^{2} $$

<br>

- 위 식에서 $$ K_{1} $$ 는 일반 상수이며 보통 0.01을 많이 사용합니다. $$ L $$ 은 픽셀값의 범위를 입력하며 일반적으로 8비트 값을 사용하여 0 ~ 255의 픽셀 값을 사용하므로 255를 $$ L $$ 로 사용합니다.
- 따라서 $$ C_{1} = (0.01 \times 255)^{2} = 6.5025 $$ 를 사용합니다.

<br>

#### **Contrast**

<br>

- `contrast`는 한글로 대조라고 하며 이미지 내에서 빛의 밝기가 바뀌는 정도를 나타내는 양입니다. 이 값은 픽셀 간의 값이 얼마나 차이가 나는 지 통하여 정량화 할 수 있으므로 표준 편차를 사용합니다.

<br>

- $$ \sigma_{x} = \left( \frac{1}{N-1} \sum_{i=1}^{N} (x_{i} - \mu_{x})^{2} \right)^{1/2} $$

- $$ N - 1 $$ : 표본의 표준 편차를 구하기 때문에 표본의 표준편차가 모표준 편차가 될 수 있도록 하기 위해 $$ N - 1 $$을 사용합니다.

- $$ \sigma_{x} $$ : 이미지의 픽셀 간 표준편차로 `contrast`를 의미합니다.

<br>

- 두 이미지의 `contrast` 성분을 비교하기 위해서는 $$ \sigma_{x} $$ 를 사용합니다. 상세식은 `luminance`의 $$ l(x, y) $$ 와 동일합니다.

<br>

- $$ c(x, y) = \frac{2\sigma_{x}\sigma_{y} + C_{2}}{\sigma_{x}^{2} + \sigma_{y}^{2} + C_{2}} \tag{2} $$

<br>

- 위 식에서도 `luminance`의 경우와 동일하게 두 이미지의 `contrast` 성분이 같을 수록 1에 가깝고 다를수록 0에 가까워집니다.
- 식 (2)의 $$ C_{2} $$ 의 경우 $$ C_{2} = (K_{2}L)^{2} $$ 로 구하며 $$ K_{2} = 0.03 $$ 을 주로 사용하여 다음과 같은 값을 가집니다.

<br>

- $$ C_{2} = (K_{2}L)^{2} = (0.03 \times 255)^{2} = 58.5225 $$

<br>

#### **Structure**

<br>

- 마지막으로 `structure`는 픽셀값의 구조적인 차이점을 나타내며 정성적으로 성분을 확인 시, edge를 나타냅니다. `sturucture`를 구하기 위하여 `luminance`을 평균, `contrast`를 표준 편차로 이용하여 Normalized된 픽셀 값의 분포에서 픽셀 값을 다시 정의합니다.

<br>

- $$ (X - \mu_{x}) / \sigma_{x} $$

- $$ X $$ : 입력 이미지

<br>

- 두 이미지의 `structure` 성분의 유사성을 확인하는 것은 두 이미지의 `correlation`을 이용하것과 같은 의미를 지닙니다.

<br>

- $$ \text{corr}(X, Y) = \frac{\sigma_{xy}}{\sigma_{x}\sigma_{y}} = \frac{E[(x - \mu_{x})(y - \mu_{y})]}{\sigma_{x}\sigma_{y}} = E \left[\frac{(x - \mu_{x})(y - \mu_{y})}{\sigma_{x}\sigma_{y}}\right] $$

- $$ = E \left[\frac{(x - \mu_{x})}{\sigma_{x}} \frac{(y - \mu_{y})}{\sigma_{y}}\right] $$

<br>

- 따라서 두 이미지의 correlation을 구하는 것은 각 이미지의 `structure` 성분의 곱의 평균을 구하는 것과 같고 `structure` 성분이 같은 방향으로 커지면 1에 가까워지는 성질을 이용하여 앞선 `luminance`, `contrast`와 동일하게 이용할 수 있습니다.
- 따라서 두 이미지의 `structure`를 비교하는 함수를 다음과 같이 정의할 수 있습니다.

<br>

- $$ s(x, y) = \frac{\sigma_{xy} + C_{3}}{\sigma_{x}\sigma_{y} + C_{3}} \tag{3} $$

- $$ \sigma_{xy} = \frac{1}{N-1} \sum_{i=1}^{N}(x_{i} - \mu_{x})(y_{i} - \mu_{y}) $$

<br>

- 이 때, $$ C_{3} $$ 는 식의 편의상 $$ C_{2} / 2 $$ 로 사용합니다. 그 이유는 `SSIM`은 $$ l(x, y), c(x, y), s(x, y) $$ 의 곱으로 정의되는 데 $$ C_{3} = C_{2} / 2 $$ 로 정의하면 식을 간편화 할 수 있습니다. 이는 이후 식을 전개하면서 보여드리겠습니다.

<br>

- 이와 같이 `luminance`, `contrast`, `strucrue`를 모두 반영한 이미지의 유사도를 결정하는 `SSIM`은 다음과 같이 정의됩니다.

<br>

- $$ \text{SSIM}(x, y) = l(x, y)^{\alpha} \cdot c(x, y)^{\beta} \cdot s(x, y)^{\gamma} \tag{4} $$

<br>

- 식 (4)에서 $$ \alpha, \beta, \gamma > 0 $$ 이면 `luminance`, `contrast`, `structure` 에 상대적인 중요도를 설정할 수 있습니다.
- 앞에서 설명한 대로 식을 간소화 하기 위하여 $$ \alpha = \beta = \gamma = 1 $$ 로 두고 $$ C_{3} = C_{2} / 2 $$ 로 하여 식을 전개해 보겠습니다.

<br>

- $$ \text{SSIM}(x, y) = l(x, y) \cdot c(x, y) \cdot s(x, y) = \frac{2\mu_{x}\mu_{y} + C_{1}}{\mu_{x}^{2} + \mu_{y}^{2} + C_{1}} \frac{2\sigma_{x}\sigma_{y} + C_{2}}{\sigma_{x}^{2} + \sigma_{y}^{2} + C_{2}} \frac{\sigma_{xy} + C_{2}/2}{\sigma_{x}\sigma_{y} + C_{2}/2} \tag{5} $$

- $$ = \frac{(2\mu_{x}\mu_{y} + C_{1}) (2\sigma_{xy} + C_{2})}{(\mu_{x}^{2} + \mu_{y}^{2} + C_{1})(\sigma_{x}^{2} + \sigma_{y}^{2} + C_{2})} \tag{6} $$

<br>

- `SSIM`은 symmetry 성질은 만족하여 x, y 이미지의 순서를 바꿔도 됩니다. 하지만 triangle inequality ( $$ \vert a + b \vert \le \vert a \vert  + \vert b \vert $$ ) 를 만족하지 않아서 distance를 구하기 위한 함수로는 사용할 수 없습니다.
- `SSIM`을 좀 더 효과적으로 사용하기 위해서는 이미지 전체를 한번에 비교하기 보다는 N x N 윈도우를 이용하여 (ex. 8 X 8, 11 X 11) 지역적으로 비교하여 사용하는 것이 효과적입니다. **왜냐하면 이미지의 왜곡이나 통계적 특성이 이미지 전반에 걸쳐서 나타나는 경우보다 지역적으로 나타나는 경우가 많고 지역적으로 더 다양한 특성을 분석할 수 있기 때문입니다.**

<br>

## **SSIM의 Pytorch에서의 사용법 (globally)**

<br>

- 먼저 이번 글에서는 

```python
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        # 입력 경계의 반사를 사용하여 상/하/좌/우에 입력 텐서를 추가로 채웁니다.
        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        # 
        # shape : (xh, xw) -> (xh + 2, xw + 2)
        x = self.refl(x) 
        # shape : (yh, yw) -> (yh + 2, yw + 2)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
```


<br>

## **SSIM의 Pytorch에서의 사용법 (locally)**

<br>


<br>


<br>



<br>


## **SSIM의 skimage에서의 사용법**

<br>

- 참조 : https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity

<br>

