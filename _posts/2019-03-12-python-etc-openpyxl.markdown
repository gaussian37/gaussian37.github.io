---
layout: post
title: openpyxl을 이용하여 python으로 엑셀 다루기
date: 2019-03-12 00:10:00 +0300
img: python/etc/openpyxl/openpyxl.jpg
categories: [python-etc] 
tags: [python, openpyxl, excel] # add tag
---

### openpyxl 설치 방법

+ openpyxl을 설치하려면 아래 명령어를 통하여 설치 합니다.

```python
pip install openpyxl
```

<br><br>

### openpyxl 사용 방법

+ 전체 매뉴얼은 다음 사이트에 있습니다.
    + https://openpyxl.readthedocs.io/en/stable/

<br>

+ openpyxl.load_workbook('파일명')을 통하여 엑셀 문서를 열 수 있습니다.
    + 이 때, open한 엑셀 파일을 객체로 받습니다.
    + ex) ```python excelFile = openpyxl.load_workbook('example.xlsx') ```

<br>

+ get_sheet_names()를 이용하여 시트의 목록을 볼 수 있습니다.
    + excelFile.get_sheet_names()

<br>

+ get_sheet_by_name('시트명')으로 특정 시트를 불러올 수 있습니다.
    + 이 때 open한 시트를 객체로 받습니다.
    + ex) ```python sheet1 = excelFile.get_sheet_by_name('Sheet1') ```

+ get_active_sheet()로 활성화된 시트를 불러올 수도 있습니다.
    + ex) ```python sheet2 = excelFile.get_active_sheet()

<br>

+ 워크시트['열행']으로 특정 셀을 불러올 수 있습니다.
    + 참고로 엑셀에서는 열/행 순서의 좌표를 가지고 있습니다.(A1은 A열 1행)
    + B1 = sheet['B1']

<br>

+ 셀 객체를 접근하면 row/column 또는 좌표자체 그리고 셀에 저장된 값을 얻을 수 있습니다.,
    + ```python B1.row ```
    + ```python B1.column ```
    + ```python B1.coordinate ```
    + ```python B1.value ```

<br>

+ 셀에 데이터를 입력하는 방법에 대하여 알아보겠습니다.
    + ```python sheet.cell(row=row_index, column=column_index).value = 값 ```
    + ```python sheet.cell(row=1, column=1).value = 10 ```
        + (1,1) 즉, A1에 10을 대입합니다.
    
