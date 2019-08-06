---
layout: post
title: 개체와 컬렉션  
date: 2019-08-06 00:00:00
img: vba/vba.png
categories: [vba-concept] 
tags: [vba, 개체, 컬렉션] # add tag
---

- 통합 문서, 워크시트, 셀, 차트 등 엑셀을 구성하는 요소를 개체(object)라고 합니다. 엑셀 프로그램도 하나의 개체입니다.
- 컬렉션(collection)은 같은 개체들의 모임입니다. 예를 들어 Sheet 개체는 하나의 시트를 의미하지만, Sheets 컬렉션은 모든 시트의 모임을 의미합니다.

<br>

### 개체 계층

- 엑셀을 사용하여 VBA 프로그램을 개발한다면 개체 모델의 가장 윗부분을 나타내는 **Application** 개체는 엑셀이 됩니다.
- **Application** 개체는 그 아래에 **Workbook, Window, Chart**등 다른 개체들을 포함하고 있고 그 개체들 또한 다른 하위 개체들을 포함하고 있습니다.
    - 예를 들어 Workbook ⊃ Worksheet ⊃ Range
- VBA 코드를 작성할 때에는 이러한 계층을 이해하고 있어야 합니다. 
- 다음은 'Test.xlsm' 워크북의 'Sheet1' 워크시트에서 'A1:C5' 범위를 참조 하기 위한 코드입니다. 상위 개체부터 하위 개체까지 마침표(.)로 구분하여 입력합니다.

```vb
Application.Workbooks("Test.xlsm").Worksheets("Sheet1").Range("A1:C5")
``` 

<br>

### 컬렉션(Collection)

- 같은 클래스에 속하는 개체 집합을 컬렉션이라고 합니다. 
- 예를 들어 Worksheets 컬렉션은 특정 Workbook 개체에 포함되어 있는 모든 워크시트 개체의 집합을 의미합니다.
- 컬렉션도 개체이므로 컬렉션을 참조해서 작업하거나, 컬렉션에 있는 특정 개체를 참조해서 작업할 수도 있습니다.
- 컬렉션에서 특정 개체를 참조할 때에는 **괄호 안에 인덱스 번호를 입력하거나 개체 이름을 입력**합니다.

```vb
Worksheets(1) : ' 첫 번째 워크시트 개체를 참조합니다.
Worksheets("Test") : ' 'Test'라는 이름의 워크시트 개체를 참조합니다.
```



