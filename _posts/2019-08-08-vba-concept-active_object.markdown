---
layout: post
title: 활성화 되어 있는 개체 참조
date: 2019-08-08 00:00:00
img: vba/vba.png
categories: [vba-concept] 
tags: [vba, 개체, 컬렉션] # add tag
---

- 여러 개의 워크북을 열어 놓고 작업 중이더라도 활성화되어 있는 워크북은 언제나 하나입니다. 하나의 워크북에 여러 개의 시트가 포함되어 있더라도 활성화되어 있는 시트는 언제나 하나입니다.
- 이와 같이 현재 활성화되어 있는 워크북, 시트, 셀 등을 간단하게 참조할 수 있는 방법을 다뤄보곘습니다.

<br>

- 현재 활성화된 워크북의 이름을 메시지 상자로 표시하려면 다음 코드로 실행할 수 있습니다.

<br>

```vb
MsgBox ActiveWorkbook.Name
```

<br>

- 워크시트에서 셀 범위가 선택되어 있다면 다음 코드는 현재 선택된 영역(Selection)에 일괄적으로 'Excel'을 입력합니다.

<br>

```vb
Selection = "Excel"
```

<br>

- 위와 같이 현재 활성화되어 있는 개체를 참조하기 위해 다음과 같이 **Application**개체의 여러 속성들을 사용합니다.
    - **ActiveCell** : 활성 셀 참조
    - **ActiveSheet** : 활성 시트 참조
    - **ActiveWindow** : 활성 창 참조
    - **ActiveWorkbook** : 활성화되어 있는 워크북 참조
    - **ThisWorkbook** : 현재 프로시저가 들어 있는 워크북 참조
    - **ActiveChar** : 활성화되어 있는 차트 참조
    - **Selection** : 선택되어 있는 개체 

