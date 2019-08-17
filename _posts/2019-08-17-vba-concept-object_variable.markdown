---
layout: post
title: 개체 변수(Object Variable)
date: 2019-08-17 00:00:00
img: vba/vba.png
categories: [vba-concept] 
tags: [vba, 개체, 개체 변수, object variable] # add tag
---

- 이번 글에서는 `개체 변수`에 대하여 간략하게 알아보도록 하겠습니다.
- 개체 변수란 워크북, 시트, 셀 범위 등 **개체를 저장할 수 있는 변수**입니다. 
- 개체 변수를 사용하면 VBA 코드를 더 단순화 시킬 수 있으며, 프로그램의 실행 속도도 더 향상시킬 수 있습니다.
- 개체에 대한 정의는 다음 글을 참조하시기 바랍니다.
    - https://gaussian37.github.io/vba-concept-object_and_collection/

<br>

```vb
Dim rng As Range '개체 변수의 선언
Set rng = Sheets(1).Range("A1:C5") '개체 변수에 개체 할당하기
rng = "HI" '개체 변수 사용
rng.Font.Size = 15 '개체 변수 사용
```

<br>

- 개체 변수를 선언하는 방법은 일반 변수를 선언하는 것과 크게 다르지 않습니다.
- 개체 변수를 선언할 때에는 데이터 형식을 저장하고자 하는 개체로 지정하면 됩니다.
    - `Dim a As Range`
    - `Dim b as Worksheet`
    - `Dim c as Workbook`

<br>

- 개체 변수에 개체를 할당하는 방법은 다른 변수에 값을 지정하는 것과 조금 다릅니다.
- 다른 변수는 '변수명 = 값' 형식으로 변수에 값을 지정하는 반면, 개체 변수에 개체를 할당하기 위해서는 `Set` 키워드를 함께 사용해야 합니다.
- 위 코드를 보면 **Set** 키워드로 개체 변수에 범위를 할당한 것을 확인할 수 있습니다.

<br>

- 개체 변수를 사용하려면 개체 변수에 지정되어 있는 속성 값에 접근해서 사용해야 합니다.
- 예를 들어 위 코드를 보면 `rng.Font.Size = 15`와 같이 **Range**개체에 속한 **Font, Size**에 접근한 것입니다.
- 반면 `rng = "HI"`를 보면 속성값에 따로 접근하지 않았는데 그 이유는 **Range**의 기본 속성 값이 **Value**이기 때문에 `rng`는 사실 `rng.Value`와 같기 때문입니다.


