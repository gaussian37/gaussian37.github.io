---
layout: post
title: openpyxl을 이용하여 python으로 엑셀 다루기
date: 2019-03-12 00:10:00 +0300
img: python/etc/openpyxl/openpyxl.jpg
categories: [python-etc] 
tags: [python, openpyxl, excel] # add tag
---

### openpyxl 설치 방법

+ openpyxl을 설치하려면 아래 명령어를 통하여 **설치** 합니다.

```python
pip install openpyxl
```

<br><br>

### openpyxl 사용 방법

+ 전체 매뉴얼은 다음 사이트에 있습니다.
    + https://openpyxl.readthedocs.io/en/stable/

<br>

+ openpyxl.load_workbook('파일명')을 통하여 엑셀 **문서를 열 수 있습니다.**
    + 이 때, open한 엑셀 파일을 객체로 받습니다.
    + ex) ```python excelFile = openpyxl.load_workbook('example.xlsx') ```

<br>

+ get_sheet_names()를 이용하여 시트의 **목록을 볼 수 있습니다.**
    + excelFile.get_sheet_names()

<br>

+ get_sheet_by_name('시트명')으로 특정 **시트를 불러올 수 있습니다.**
    + 이 때 open한 시트를 객체로 받습니다.
    + ex) ```python sheet1 = excelFile.get_sheet_by_name('Sheet1') ```

+ get_active_sheet()로 활성화된 시트를 불러올 수도 있습니다.
    + ex) ```python sheet2 = excelFile.get_active_sheet()

<br>

+ 워크시트['열행']으로 **특정 셀을 불러올 수 있습니다.**
    + 참고로 엑셀에서는 열/행 순서의 좌표를 가지고 있습니다.(A1은 A열 1행)
    + B1 = sheet['B1']

<br>

+ 셀 객체를 접근하면 row/column 또는 좌표자체 그리고 **셀에 저장된 값을 얻을 수 있습니다.**
    + ```python B1.row ```
    + ```python B1.column ```
    + ```python B1.coordinate ```
    + ```python B1.value ```

<br>

+ 셀에 **데이터를 입력**하는 방법에 대하여 알아보겠습니다.
    + ```python sheet.cell(row=row_index, column=column_index).value = 값 ```
    + ```python sheet.cell(row=1, column=1).value = 10 ```
        + (1,1) 즉, A1에 10을 대입합니다.

<br>

+ 엑셀 파일 **새로 만들어서 저장**하는 방법

```python
filepath = "/test.xlsx"
wb = openpyxl.Workbook()

wb.save(filepath)
```

<br>

+ sheet.iter_rows()와 sheet.iter_cols()를 이용하여 **특정 범위의 셀에 접근**할 수 있습니다.

```python
allList = []
for row in sheet.iter_rows(min_row=1, max_row=10, min_col=2, max_col=5):
    a = []
    for cell in row:
        a.append(cell.value)
    allList.append(a)
```

+ sheet.iter_rows()/iter_cols()를 선언하면 generator가 생성됩니다.
+ 생성된 generator를 이용하여 row와 col의 min/max 범위에 맞게 접근합니다.

<br>

+ 엑셀 함수를 사용하다보면 열/행조합(ex. A1:A7)으로 범위를 접근하는 경우가 있습니다.
+ openpyxl에서도 다음과 같이 범위를 이용하여 셀에 접근할 수 있습니다.
    + 특정 범위 접근
        + cell_range = sheet['A1':'C2']
	+ 특정 row 접근
	    + row10 = sheet[10]
    + 특정 row 범위 
	    + row_range = sheet[5:10]
	+ 특정 Column
	    + colC = sheet['C']
	+ 특정 Column 범위
	  - col_range = sheet['C:D']
	  
<br>

+ 엑셀의 행 또는 열을 추가하려면 다음과 같이 추가합니다.
    + 함수의 인자로 입력된 숫자 앞에 행 또는 열이 추가됩니다.
    + sheet.insert_cols(숫자)
    + sheet.insert_rows(숫자)
