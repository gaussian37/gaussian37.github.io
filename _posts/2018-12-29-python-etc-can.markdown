---
layout: post
title: python으로 CAN 데이터 읽기 
date: 2018-12-19 13:46:00
img: python/etc/can/can.jpg
categories: [python-etc] 
tags: [python, can, mdf] # add tag
---

<br>

- 이번 글에서는 python을 이용하여 CAN 데이터를 저장하는 방법에 대하여 간략하게 알아보겠습니다.
- 정확히는 `MDF(Measurement Data Format)`로 저장된 데이터를 읽는 방법이라고 할 수 있습니다.
    - 참고로 `MDF`는 벡터와 보쉬가 자동차 산업에서 사용하기 위해 만든 계측 데이터 용도의 바이너리 파일입니다.
    - 참조 : https://www.vector.com/kr/ko/products/application-areas/ecu-calibration/measurement/mdf/
- 일반적으로 많이 사용하는 `Vector` 회사의 장비를 이용하여 CAN 데이터를 수집한다면 `MDF`파일로 CAN 값을 저장할 수 있습니다.
    - CAN Logger 장비(GL 시리즈), CANoe, CANalyzer 등은 모두 지원합니다.
- MDF 파일로 저장할 때, 반드시 `CAN DB`를 CAN 데이터와 연동하여 저장해야 파이썬에서 읽을 때 CAN에 대한 설명값을 얻을 수 있습니다.
    - CAN DB를 연동하는 것은 벡터 소프트웨어에서 `MDF` 형식으로 저장할 때 할 수 있습니다.  

<br>

## **필요한 라이브러리 설치**

- `asammdf` 공식 라이브러리 페이지 : https://pypi.org/project/asammdf/
-- `pip install asammdf` 을 통하여 asammdf를 설치할 수 있습니다. 그 전에 아래 필수 라이브러리와 옵션 라이브러리를 먼저 설처히시기 바랍니다.
- 필수 라이브러리    
    - `pip install twisted`
    - `pip install numpy` : the heart that makes all tick
    - `pip install numexpr` : for algebraic and rational channel conversions
    - `pip install wheel` : for installation in virtual environments
    - `pip install pandas` : for DataFrame export
    - `pip install canmatrix` : to handle CAN bus logging measurements
    - `pip install natsort` 
    - `pip install cChardet` : to detect non-standard unicode encodings
    - `pip install lxml` : for canmatrix arxml support
- 옵션 라이브러리
    - `pip install h5py` : for HDF5 export
    - `pip install scipy` :  for Matlab v4 and v5 .mat export
    - `pip install hdf5storage` : for Matlab v7.3 .mat export
    - `pip install fastparquet` : for parquet export
    - `pip install PyQt5` : for GUI tool 
    - `pip install pyqtgraph` : for GUI tool and Signal plotting (preferably the latest develop branch code)
    - `pip install matplotlib` : as fallback for Signal plotting 

<br>

- 참고로 `ASAM`은 ASAM (Association for Standardisation of Automation-and Measuring System)의 약자로 차량용 제어기 개발에 대한 시간과 비용을 줄이기 위해 설립된 단체입니다.

<br>

## **CAN 데이터 읽고 실행하는 방법**

<br>

- [실습 예제 파일](https://drive.google.com/open?id=1kbmElexO_jwdm60WXp_lOsUQDTsTWXvz)
    - 벡터 사에서 제공하는 데모 소프트웨어 파일로 벡터사 홈페이지에서도 받을 수 있습니다.
- MDF 파일을 불러오는 방법은 아래 코드와 같습니다. `asammdf`의 MDF를 이용하여 파일을 불러올 수 있습니다.
- MDF 파일 내의 CAN 신호 리스트는 `.channels_db`에서 확인 가능합니다. 

<br>

```python
from asammdf import MDF, Signal
import pandas as pd
import numpy as np
import matplotlib
# import matplotlib.pyplot as plt

# MDF 파일을 읽어옵니다.
path = "MDF 파일을 저장한 경로"
data = MDF(path + "Acceleration_StandingStart.MDF")

### CAN 신호 리스트를 가져 옵니다.
signal_list = list(data.channels_db)
# 가져온 리스트에서 시간축은 신호가 아니므로 제외합니다.
signal_list.remove('t')
print(signal_list) # 로깅된 CAN 신호 전체를 볼 수 있습니다.

### 그래프 출력
speed = data.get('VehicleSpeed')
speed.plot()

### 여러 그래프 출력
for signal in data.select(filtered_signal_list):
    signal.plot()

### 필요한 신호만 필터링
filtered_signal_list = ['VehicleSpeed', 'Throttle']
# 10초 ~ 12초 사이의 데이터만 필터링
filtered_data = data.filter(filtered_signal_list).cut(start=10, stop=12)

### 엑셀 파일 또는 CSV 파일로 출력
signals_data_frame = data.to_dataframe()
signals_data_frame.to_excel(path + "signals_data_frame.xlsx")
signals_data_frame.to_csv(path + "signals_data_frame.csv")
```

<br>

- 위 결과에 나타난 `key` 값이 CAN 신호값에 해당합니다. 그리고 `t`는 계측된 전체 시간에 해당합니다.
- CAN 신호는 dictionary 형태로 저장되어 있으므로 (key, value)로 값 접근이 가능합니다.
- `.channels_db`에서 확인한 값을 Key 값으로 하면 각 신호의 값을 알 수 있습니다.
- `t`를 Key 값으로 하면 신호가 기록된 시간을 알 수 있습니다.
- 따라서 `t`에 해당하는 신호값을 찾으면 샘플링 시간에 기록된 신호값을 확인할 수 있습니다.
- 아래 코드는 x축을 시간, y축을 속도로 나타낸 차속 그래프 입니다.

<br>
<center><img src="../assets/img/python/etc/can/2.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- CAN 신호값을 파이썬으로 불러왔기 때문에 엑셀로 저장도 가능합니다.
- 아래 코드는 Pandas를 이용하여 처리하기 좋은 엑셀로 저장하는 코드 입니다.

<br>
<center><img src="../assets/img/python/etc/can/3.png" alt="Drawing" style="width: 1000px;"/></center>
<br>
