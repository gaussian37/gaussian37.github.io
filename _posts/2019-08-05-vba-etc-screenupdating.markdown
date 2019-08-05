---
layout: post
title: Screen updating 기능  
date: 2019-07-23 00:00:00
img: vba/vba.png
categories: [vba-etc] 
tags: [vba, screen updating] # add tag
---

- VBA를 사용하여 작업을 수행 할 때, 실제 화면에서 변경되어야 하는 양이 많으면 VBA 명령어를 수행함과 동시에 실제 화면 변경이 필요해서 작업이 오래 걸립니다.
- 이럴 때에는 보통 모든 작업이 끝나면 한번에 결과를 표시하도록 하면 작업 수행 속도가 빨라집니다.
- 사실상 실제 사람들이 사용하도록 배포할 때에는 반드시 해놓아야 하는 기능입니다. 

```vb

' Screen update 기능을 꺼서 화면에 변경사항이 발생하여도 반영 하지 않도록 합니다. 
Application.ScreenUpdating = False

' ....
' Source Code
' ....

' 코드의 마지막에 Screen update 기능을 다시 켜는 코드를 삽입하여 설정을 원상 복귀 시킵니다.
Application.ScreenUpdating = True

```





