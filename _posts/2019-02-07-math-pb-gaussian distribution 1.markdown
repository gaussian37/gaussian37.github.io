---
layout: post
title: 이변량 정규 분포 (1)
date: 2019-02-07 00:00:00
img: math/pb/probability.jpg
categories: [math-pb] 
tags: [통계학, 수리통계학, 이변량 정규분포, 정규분포] # add tag
---

+ 출처 : 
    + Probability & Statistics for Engineers & Scientists. 9th Edition.(Walpole 저. PEARSON) 
    + 수리통계학 (김수택 저. 자유 아카데미)

+ [통계학 관련 글 목록](https://gaussian37.github.io/math-pb-probability-table/)

+ 이번 글과 다음글에서는 `이변량 정규 분포`에 대하여 자세하게 알아보도록 하겠습니다.

<br><br>

## 조건부 평균과 조건부 분산

+ X, Y의 결합 밀도함수가 $$ f(x, y) $$이고, 각각의 주변분포 함수를 $$ f_{x}(x), f_{y}(y) $$ 라고 합니다.
    + 이산형에서 주변 분포를 구할 때, 어떤 x를 기준으로 y를 모두 더하면 x에 대한 주변 분포를 구할 수 있습니다.
    + 이산형에서 주변 분포를 구할 때, 어떤 y를 기준으로 x를 모두 더하면 y에 대한 주변 분포를 구할 수 있습니다.
    + 연속형에서는 주변 분포를 구할 때, 어떤 x를 기준으로 y에 대하여 적분하면 x에 대한 주변 분포를 구할 수 있습니다.
    + 연속형에서는 주변 분포를 구할 때, 어떤 y를 기준으로 x에 대하여 적분하면 y에 대한 주변 분포를 구할 수 있습니다.

<br>

+ 조건부 확률 밀도 함수
    + X = x로 주어질 때, Y에 대한 조건부 확률밀도 함수 : $$ f_{Y \vert X}(y \vert x) = \frac{f(x,y)}{f_{X}(x)} $$
        + 즉, x가 고정일 때 Y의 분포
    + Y = y로 주어질 때, X에 대한 조건부 확률밀도 함수 : $$ f_{X \vert Y}(x \vert y) = \frac{f(x,y)}{f_{Y}(y)} $$
        + 즉, y가 고정일 때 X의 분포

<br>
        
+ 조건부 평균
    + X = x 일 때, Y의 조건부 평균 (조건부 기댓값)
        + ·$$ \mu_{Y \vert X} = \mathcal E(Y \vert x) $$
            + 이산형 : $$ \sum_{y} y f(y \vert x) $$
            + 연속형 : $$ \int_{-\infty}^{\infty} yf(y\vert x) dy$$
    + Y = y 일 때, X의 조건부 평균 또한 위의 방식과 똑같이 적용할 수 있습니다.
    + 구하려는 분포의 식이 x와 y의 합성 함수라고 하여도 x 또는 y가 고정되어 있으므로 같은 방식으로 적용할 수 있습니다.
    + 즉, X = x 일 때, u(x, Y)의 조건부 평균 : 
        + ·$$ \mathcal E(u(X,Y) \vert x) $$
            + 이산형 : $$ \sum_{y} u(x,y) f(y \vert x) $$
            + 연속형 : $$ \int_{-infty}^{\infty} u(x,y) f(y \vert x) dy $$
    + 조건부 평균의 핵심은 **특정 변수가 고정된 상태**에서의 평균이라고 생각하면 됩니다.

<br>

+ 조건부 분산
    + ·$$ \sigma_{Y \vert x}^{2} = E((Y - E(Y \vert x) )^{2} \vert x) = E(Y^{2} \vert x) - E(Y \vert x)^{2} $$
        + 조건부 분산의 핵심 또한 **특정 변수가 고정된 상태 에서의 분산**이라고 생각하면 됩니다.

<br>

+ 조건부 기대값의 성질
    + ·$$ E(aX + bY + c \vert Z = z) = aE(x \vert z) + bE(Y \vert z) + c $$
    + ·$$ E(E(Y \vert X)) = E(Y) $$
        + X = x로 고정된 상태에서 Y에 대한 평균을 구하는 것
        + 즉, $$ X = {x_{1}, x_{2}, ...} $$ 에서 여러가지 $$ x_{i} $$ 로 변경해 가면서 평균을 구하였을 때, 결국에는 $$ E(Y) $$ 와 같아집니다.
        + **아래 증명을 참조 하시기 바랍니다.**         
    + ·$$ E(E(u(X,Y) \vert X)) = E(u(X,Y)) $$
        + **아래 증명을 참조 하시기 바랍니다.**
    + ·$$ E(g(x)Y \vert X = x) = g(x)E(Y \vert x) $$
        + X = x로 고정되어 있기 때문에 g(x) 즉 소문자 x가 대입된 값이 곱 형태로 빠져나올 수 있습니다.
    + ·$$ \sigma^{2}_{Y} = Var(Y) = E(Var(Y \vert X)) + Var(E(Y \vert X)) $$
        + 분산은 조건부 분산의 평균과 조건의 평균의 분산의 합으로 표현 가능합니다.         
        + **아래 증명을 참조 하시기 바랍니다.**

<br>    

+ ·$$ E(E(Y \vert X)) = E(Y) $$ 증명
+ ·$$ E(E(Y \vert X)) $$
    + ·$$ = \int_{-\infty}^{\infty} (\int_{-\infty}^{\infty} y f(y \vert x) dy) f_{x}x dx $$
        + 이중 적분으로 식을 정리해 보겠습니다.
    + ·$$ = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} y f(y \vert x) f_{x}x dy dx $$
        + ·$$ f(x,y) = f(y \vert x) f(x) $$ 를 이용하여 식을 정리하면
    + ·$$ = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} y f(x,y) dx dy $$
        + 먼저 x에 대하여 적분할 수 있도록 식을 변경합니다.
    + ·$$ = \int_{-\infty}^{\infty} y(\int_{-\infty}^{\infty} f(x,y) dx) dy $$
        + x에 대하여 적분을 하면 Y 확률 변수에 대한 주변 확률 분포를 구할 수 있습니다.
    + ·$$ = \int_{-\infty}^{\infty} y f_{Y}(y) dy $$
        + y의 평균 식으로 변경이 됩니다.
    + ·$$ = E(Y) $$
     
<br>

+ ·$$ E(E(u(X,Y) \vert X)) = E(u(X,Y)) $$ 증명
+ ·$$ E(E(u(X,Y) \vert X)) $$
    + ·$$ = \int_{-\infty}^{\infty}(\int_{-\infty}^{\infty} u(x,y) f(y \vert x) dy) f_{x}(x) dx $$
        + 이중 적분으로 식을 정리해 보겠습니다.
    + ·$$ = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} u(x,y) f(y \vert x) f_{x}(x) dy dx $$
        + 조건부 분포와 확률의 곱은 결합 확률 분포로 나타낼 수 있습니다.
    + ·$$ = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} u(x,y) f(x,y) dy dx $$
    + ·$$ = E(u(X, Y)) $$
        
<br>

+ ·$$ \sigma^{2}_{Y} = Var(Y) = E(Var(Y \vert X)) + Var(E(Y \vert X)) $$ 증명
+ ·$$ E(Var(Y \vert X)) $$
+ ·$$ = E(E(Y^{2} \vert  X) - E(Y \vert X)^{2}) $$
    + ·$$ E(E(Y^{2} \vert  X) = E(Y^{2}) $$ 이므로
+ ·$$ = E(Y^{2}) - E(Y)^{2} + E(Y)^{2} - E(Y \vert X)^{2}) $$
    + 식의 전개를 위하여 $$ - E(Y)^{2} + E(Y)^{2} $$ 을 추가하였습니다.
    + ·$$ Var(Y) = E(Y^{2}) - E(Y)^{2} $$ 식을 다음 전개에 이용하겠습니다.
    + ·$$ E(E(Y \vert X)) = E(Y) $$ 식을 다음 전개에 이용하겠습니다.
+ ·$$ = Var(Y) + E(E(Y \vert X))^{2} - E(E(Y \vert X)^{2})
+ ·$$ = Var(Y) - Var(E(Y \vert X))
+ 식을 최종적으로 정리하면 $$ Var(Y) = E(Var(Y \vert X)) + Var(E(Y \vert X)) $$ 가 됩니다.

<br>

+ 공분산
    + ·$$ Cov(X, Y) = \sigma_{XY} = E( (X-\mu_{X})(Y-\mu_{Y}) ) $$
        + 이산형 : $$ \sum_{x}\sum_{y} (x - \mu_{x})(y - \mu_{y}) f(x, y) $$
        + 연속형 : $$ \int_{-\infty}^{\infty} y(\int_{-\infty}^{\infty} (x - \mu_{x})(y - \mu_{y}) f(x, y) dx dy $$
    + ·$$ \sigma_{XY} = E(XY) - \mu_{X}\mu_{Y} $$
+ 상관계수
    + ·$$ \rho_{XY} = \frac{\sigma_{XY}}{\sigma_{X}\sigma_{Y}} = \frac{Cov(X, Y))}{ \sqrt{Var(X)}\sqrt{Var(Y)} } $$
+ 공분산과 상관계수를 이용하여 조건부 평균에 이용해 보겠습니다.

<br>

+ X = x일 때, Y의 **조건부 평균**이 x에 대해 선형인 경우 (즉, $$ E(Y \vert x) = a + bx $$)
+ 이산형의 경우 $$ a + bx = E(Y \vert x) = \sum_{y} yf(y \vert x) = \sum_{y} y \frac{ f(x, y) }{ f_{X}(x) } $$
    + ·$$ \sum_{y} y f(x, y) = (a + bx)f_{X}(x) $$
        + 양변에 summation x를 추가해 줍니다.
    + ·$$ \sum_{x}\sum_{y} yf(x,y) = \sum_{x}(a + bx)f_{x}(x) $$
        + 좌변은 $$ E(Y) $$로 우변은 $$ E(a + bx) $$로 정리 됩니다. 
    + ·$$ E(Y) = E(a + bx) $$ 
    + ·$$ \mu_{Y} = a + b\mu_{X} $$ ---① 식
    



  


    
            
        
                   
     




