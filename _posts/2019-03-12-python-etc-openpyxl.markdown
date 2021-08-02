---
layout: post
title: openpyxl을 이용하여 python으로 엑셀 다루기
date: 2019-03-12 00:10:00
img: python/etc/openpyxl/openpyxl.jpg
categories: [python-etc] 
tags: [python, openpyxl, excel] # add tag
---

<br>

- 참조 : https://openpyxl.readthedocs.io/en/stable/#

<br>

## **목차**

<br>

- ### [openpyxl 설치 방법](#openpyxl-설치-방법-1)
- ### [openpyxl 기본 사용법](#openpyxl-기본-사용법-1)
- ### [pandas to openpyxl](#pandas-to-openpyxl-1)
- ### [openpyxl to pandas](#openpyxl-to-pandas-1)
- ### [비밀번호로 시트 보호](#비밀번호로-시트-보호-1)
- ### [열 너비 자동 맞춤](#열-너비-자동-맞춤-1)

<br>

## **openpyxl 설치 방법**

<br>

- openpyxl을 설치하려면 아래 명령어를 통하여 **설치** 합니다.

<br>

```python
pip install openpyxl
```

<br>

## **openpyxl 기본 사용법**

<br>

- 전체 매뉴얼은 다음 사이트에 있습니다.
    - https://openpyxl.readthedocs.io/en/stable/

<br>

- `openpyxl.load_workbook('파일명')`을 통하여 엑셀 **문서를 열 수 있습니다.**
- 이 때, open한 엑셀 파일을 객체로 받습니다.
- ex) `excelFile = openpyxl.load_workbook('example.xlsx')`

<br>

- 현재 활성 중인 워크시트를 선택하는 방법은 다음과 같습니다.
- `sheet = wb.active`

<br>

- 또는 sheet의 이름을 직접 입력하여 sheet를 선택할 수 있습니다.
- `sheet = wb["SheetName"]`

<br>

- `get_sheet_names()`를 이용하여 시트의 **목록을 볼 수 있습니다.**
- `excelFile.get_sheet_names()`

<br>

- `get_sheet_by_name('시트명')`으로 특정 **시트를 불러올 수 있습니다.**
- 이 때 open한 시트를 객체로 받습니다.
- ex) `sheet1 = excelFile.get_sheet_by_name('Sheet1')`

<br>

- `get_active_sheet()`로 활성화된 시트를 불러올 수도 있습니다.
- ex) `sheet2 = excelFile.get_active_sheet()`

<br>

- 워크시트['열행']으로 **특정 셀을 불러올 수 있습니다.**
- 참고로 엑셀에서는 열/행 순서의 좌표를 가지고 있습니다.(A1은 A열 1행)
- ex) `B1 = sheet[B1]`

<br>

- 어떤 sheet의 이름을 변경하고 싶으면 다음과 같이 바꿀 수 있습니다.
- `sheet.title = new title`

<br>

- 셀 객체를 접근하면 row/column 또는 좌표자체 그리고 **셀에 저장된 값을 얻을 수 있습니다.**
- ex) `B1.row`, `B1.column`, `B1.coordinate`, `B1.value`

<br>

- 셀에 **데이터를 입력**하는 방법에 대하여 알아보겠습니다.
- `sheet.cell(row=row_index, column=column_index).value = 값`
- ex) `sheet.cell(row=1, column=1).value = 10` : (1,1) 즉, A1에 10을 대입합니다.

<br>

- 엑셀 파일 **새로 만들어서 저장**하는 방법

<br>

```python
filepath = "/test.xlsx"
wb = openpyxl.Workbook()
wb.save(filepath)
```

<br>

- sheet.iter_rows()와 sheet.iter_cols()를 이용하여 **특정 범위의 셀에 접근**할 수 있습니다.

<br>

```python
allList = []
for row in sheet.iter_rows(min_row=1, max_row=10, min_col=2, max_col=5):
    a = []
    for cell in row:
        a.append(cell.value)
    allList.append(a)
```

<br>

- sheet.iter_rows()/iter_cols()를 선언하면 generator가 생성됩니다.
- 생성된 generator를 이용하여 row와 col의 min/max 범위에 맞게 접근합니다.

<br>

- 엑셀 함수를 사용하다보면 열/행조합(ex. A1:A7)으로 범위를 접근하는 경우가 있습니다.
- openpyxl에서도 다음과 같이 범위를 이용하여 셀에 접근할 수 있습니다.
    - 특정 범위 접근
        - cell_range = sheet['A1':'C2']
	- 특정 row 접근
	    - row10 = sheet[10]
    - 특정 row 범위 
	    - row_range = sheet[5:10]
	- 특정 Column
	    - colC = sheet['C']
	- 특정 Column 범위
	  - col_range = sheet['C:D']
	  
<br>

- 엑셀의 행 또는 열을 추가하려면 다음과 같이 추가합니다. 함수의 인자로 입력된 숫자 앞에 행 또는 열이 추가됩니다.
- `sheet.insert_cols(숫자)`
- `sheet.insert_rows(숫자)`

<br>

- 셀에 `음영` 색깔 칠하려면 색 정보를 `PatternFill`을 통해 생성한 다음에 셀에 `fill` 해주면 됩니다.

<br>

```python
from openpyxl.styles import PatternFill

# 음영 색 지정
yellowFill = PatternFill(start_color='FFFFFF00',
                   end_color='FFFFFF00',
                   fill_type='solid')

# 지정된 음영 색으로 음역 색칠하기
sheet["A1"].fill = yellowFill		   
```

<br>

- 새로운 sheet를 추가하려면 workbook 객체에서 `.create_sheet(title=None, index=None)` 함수를 사용하면 됩니다.
- `wb.create_sheet(title=None, index=None)`와 같은 형태이며 예를 들어 `wb.create_sheet(index = 1 , title = sheetname)`와 같이 사용할 수 있습니다.
- 두 인자 모두 선택적으로 사용가능하며 모두 입력하지 않으면 기본값으로 입력됩니다.

<br>

## **pandas to openpyxl**

<br>

- 파이썬으로 structured data를 다룰 때 가장 많이 사용하는 자료형이 `pandas` 입니다. pandas를 사용하면 굉장히 편리하게 structured data를 다룰 수 있기 때문입니다. 따라서 openpyxl에서 모든 작업을 다 처리하는 것 보다 `pandas`에서 작업을 하고 마지막에 openpyxl을 이용하여 xlsx 포맷으로 변경하는 것이 편리합니다.
- 따라서 이번 글에서는 pandas를 openpyxl로 변경하는 방법을 살펴보겠습니다. 먼저 가장 간단하게 변환할 수 있는 방법은 아래와 같습니다. active된 worksheet에 행 방향으로 데이터가 쌓입니다.

<br>

```python
from openpyxl.utils.dataframe import dataframe_to_rows
wb = Workbook()
ws = wb.active

for r in dataframe_to_rows(df, index=True, header=True):
    ws.append(r)
```

<br>

- 만약 `header`와 `index`를 pandas 타입과 같이 강조하려면 다음과 같이 사용하시면 됩니다. 한번 출력해 보시면 내용을 이해하실 수 있습니다.

<br>

```python
wb = Workbook()
ws = wb.active

for r in dataframe_to_rows(df, index=True, header=True):
    ws.append(r)

for cell in ws['A'] + ws[1]:
    cell.style = 'Pandas'

wb.save("pandas_openpyxl.xlsx")
```

<br>

## **openpyxl to pandas**

<br>

- 이번에는 앞의 예제와 반대로 기존의 `엑셀 파일`을 `pandas`로 변경하는 방법에 대하여 살펴보겠습니다.
- 물론 엑셀 파일의 내용은 pandas로 변경이 가능한 내용이 저장되어 있어야 합니다.

<br>

```python
import openpyxl
from itertools import islice
from pandas import DataFrame

wb = openpyxl.load_workbook('example.xlsx')
ws = wb.active

# 모든 값을 DataFrame에 포함하고 index와 comlumn 이름은 0, 1, 2, ...의 숫자 사용
df = DataFrame(ws.values)

data = ws.values
cols = next(data)[1:]
data = list(data)
idx = [r[0] for r in data]
data = (islice(r, 1, None) for r in data)

# 첫 행이 index이고 첫 열이 column 명일 경우에 사용
df = DataFrame(data, index=idx, columns=cols)
```

<br>

## **비밀번호로 시트 보호**

<br>

- 참조 : https://openpyxl.readthedocs.io/en/stable/protection.html
- 엑셀 파일을 보호하고 수정을 방지하기 위하여 `protection` 기능을 사용할 수 있습니다. 엑셀 파일 중 보호하는 대상은 `workbook`과 `worksheet` 입니다.
- `workbook`에 protection 기능을 적용하면 엑셀 시트 이외의 전체 구조에 대한 protection을 적용할 수 있습니다.
- 반면 `worksheet`에 protection 기능을 적용하면 워크시트에 입력된 값 및 서식들에 대하여 protection 기능을 적용할 수 있습니다.
- 아래 코드에서 `...` 라고 되어 있는 패스워드 란을 실제 사용할 패스워드로 바꿔서 사용하면 됩니다.

<br>

```python
# workbook
wb.security.workbookPassword = '...'
wb.security.lockStructure = True


ws = wb.active
ws.protection.sheet = True
ws.protection.enable()
ws.protection.password = '...'
# ws.protection.disable()
```

<br>

## **열 너비 자동 맞춤**

<br>

- 엑셀에서 열 너비 자동 맞춤 기능은 일괄적으로 시트를 보기 좋게 만들기 위하여 종종 사용합니다.
- 아래 `AutoFitColumnSize` 함수를 사용하면 원하는 열 또는 모든 열에 대하여 열 너비를 자동 맞춤하는 기능을 적용할 수 있습니다.
- 이 때, 적용되는 열의 너비는 열의 각 셀 중 가장 긴 문자를 포함하는 셀의 너비에 margin을 더한 값입니다. 1 이상의 margin을 가져야 빽빽하지 않게 자동 맞춤이 됩니다.
- 아래 함수에서 `columns`는 리스트를 받고 따로 `None` 또는 입력을 하지 않으면 전체 열을 대상으로 자동 맞춤을 적용하여 `[1, 2, 3]`과 같이 적용하면 1열, 2열, 3열만 자동 맞춤을 적용하게 됩니다.

<br>

```python
# culumns is passed by list and element of columns means column index in worksheet.
# if culumns = [1, 3, 4] then, 1st, 3th, 4th columns are applied autofit culumn.
# margin is additional space of autofit column. 

def AutoFitColumnSize(worksheet, columns=None, margin=2):
    for i, column_cells in enumerate(worksheet.columns):
        is_ok = False
        if columns == None:
            is_ok = True
        elif isinstance(columns, list) and i in columns:
            is_ok = True
            
        if is_ok:
            length = max(len(str(cell.value)) for cell in column_cells)
            worksheet.column_dimensions[column_cells[0].column_letter].width = length + margin

    return worksheet
```

<br>