---
layout: post
title: VBA에서 사용 가능한 함수들
date: 2019-08-11 00:00:00
img: vba/vba.png
categories: [vba-concept] 
tags: [vba, 함수, function, 워크시트 함수, VBA 내장 함수, 프로시저] # add tag
---

- 이번 글에서는 VBA에서 사용 가능한 함수들에 대하여 다루어 보도록 하겠습니다.
- 먼저 VBA에서 사용 가능한 함수는 크게 3가지 입니다.
    - 엑셀 워크시트 함수
    - VBA 내장 함수
    - 프로시저
    
<br>

## 엑셀 워크시트 함수

<br>

- 먼저 엑셀의 워크시트 함수부터 알아보도록 하겠습니다.
- VBA에서 사용할 수 있는 엑셀의 워크시트 함수는 전체를 모두 사용하는 것은 아니고 일부만 사용할 수 있습니다.
- 그 리스트는 다음 링크를 확인해 보시길 바랍니다.
    - [전체 리스트](https://docs.microsoft.com/en-us/office/vba/excel/concepts/events-worksheetfunctions-shapes/list-of-worksheet-functions-available-to-visual-basic)
    - [카테고리 별 리스트](https://support.office.com/en-us/article/excel-functions-by-category-5f91f4e9-7b42-46d2-9bd1-63f26a86c0eb?ui=en-US&rs=en-US&ad=US)
- 사람들이 가장 많이 참조했다는 TOP10 워크시트 함수는 다음과 같습니다.
    - [SUM 함수]()https://support.office.com/en-us/article/sum-function-043e1c7d-7726-4e80-8f32-07b23e057f89)
    - [IF 함수](https://support.office.com/en-us/article/if-function-69aed7c9-4e8a-4755-a9bc-aa8bbff73be2) 
    - [LOOKUP 함수](https://support.office.com/en-us/article/lookup-function-446d94af-663b-451d-8251-369d5e3864cb)
    - [VLOOKUP 함수](https://support.office.com/en-us/article/vlookup-function-0bbc8083-26fe-4963-8ab8-93a18ad188a1)
    - [MATCH 함수](https://support.office.com/en-us/article/match-function-e8dffd45-c762-47d6-bf89-533f4a37673a)
    - [CHOOSE 함수](https://support.office.com/en-us/article/choose-function-fc5c184f-cb62-4ec7-a46e-38653b98f5bc)
    - [DATE 함수](https://support.office.com/en-us/article/date-function-e36c0c8c-4104-49da-ab83-82328b832349)
    - [DAYS 함수](https://support.office.com/en-us/article/days-function-57740535-d549-4395-8728-0f07bff0b9df)
    - [FIND, FINDB 함수](https://support.office.com/en-us/article/find-findb-functions-c7912941-af2a-4bdf-a553-d0d89b0a0628)
    - [INDEX 함수](https://support.office.com/en-us/article/index-function-a5dcf0dd-996d-40a4-a822-b56b061328bd)
- 워크시트 함수를 사용하는 방법은 크게 2가지 방법이 있습니다.
    - 워크시트 함수를 이용해 계산된 값을 셀에 기록하는 예
        - `Range("B12").Formula = "=AVERAGE(B2:B11)"`
        - `Range("B12").FormulaR1C1 = "=AVERAGE(R[-10]C:R[-1]C)"`   
    - VBA에서 워크시트 함수 이용하기
        - `my_sum = Application.WorksheetFunction.SUM(B2:D2)`
        - `my_sum = Application.WorksheetFunction.AVERAGE(var1 + var2)`
            - 워크 시트 함수의 인수에 변수를 넣을 수 있습니다.

