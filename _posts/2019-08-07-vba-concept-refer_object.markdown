---
layout: post
title: 개체 참조하기  
date: 2019-08-07 00:00:00
img: vba/vba.png
categories: [vba-concept] 
tags: [vba, 개체, 컬렉션] # add tag
---

- VBA 코드를 통해 어떤 셀에 값을 입력하거나 서식을 지정하는 등의 작업을 하고 싶다면 작업의 대상이 되는 셀이 어떤 셀인지 정확하게 참조할 수 있어야 합니다.
- 가장 많이 다루는 개체 중 `Range` 개체로 셀을 참조하는 예를 통하여 개체 참조에 대하여 알아보겠습니다.

<br>

- VBA에서 개체를 참조하려면 상위 개체와 하위 개체를 마침표(.)로 연결합니다.
- 다음 코드는 **Sample.xlsm** 워크북의 **Sheet1** 시트에서 **A1**셀을 참조하여 100을 입력합니다.

<br>

```vb
Application.Workbooks("Sample.xlsm").Worksheets("Sheet1").Range("A1") = 100
```

<br>

- 특별한 경우를 제외하고 최상위 개체인 **Application** 개체는 생략해도 됩니다.
- 또 현재 워크북에 있는 워크시트를 참조한다면 **Workbook**에 대한 참조도 생략이 가능합니다.
- 같은 의미로 현재 워크북의 현재 시트에 있는 셀을 참조한다면 **Worksheet**에 대한 참조도 생략할 수 있습니다.
- Range 개체는 셀이나 셀 범위를 의미합니다.

<br>

```vb
Worksheets("Sheet1").Range("A1") = 100 ' Sheet1 워크시트의 [A1] 셀에 100을 입력
Range("A1") = 100 ' 현재 워크시트의 A1 셀에 100을 입력
```

<br>

- 개체를 참조하기 전에 상위 개체를 먼저 선택(Select)하는 방법도 자주 사용됩니다.
- 다음 코드는 개체를 선택하는 Select 메서드를 통하여 **Sheet1** 워크시트를 선택한 다음 Range 개체로 셀을 참조하여 값을 입력하는 예입니다.

<br>

```vb
Worksheets("Sheet1").Select 'Sheet1 워크시트 선택
Range("A1") = 100 ' A1 셀에 100 입력
```

<br>