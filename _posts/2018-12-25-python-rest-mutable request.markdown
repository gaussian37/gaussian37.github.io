---
layout: post
title: django restframework에서 create 시 입력 내용 수정
date: 2018-12-25 18:43:00
img: python/rest/rest.png
categories: [python-rest] 
tags: [python, django, REST, request, mutable] # add tag
---

POST(create)를 할 때, 사용자로 부터 입력 받은 값을 이용하여 처리를 하고 싶을 땐 어떻게 하면 좋을까요?

예를 들어 사용자로 부터 위도/경도 좌표 값을 입력 받았을 때, 특정 지점으로 부터 거리를 구하고 싶을 수 있습니다.
이 때 사용자가 특정 지점의 위도/경도를 입력하면, POST 하는 그 시점에 자동으로 거리 field에 입력되도록 하고 싶습니다.

그러면 `django-restframework`의 `viewset` 에서 `create` 함수를 통하여 처리할 수 있습니다.
`viewset`의 `create` 함수는 POST 기능을 지원하도록 상속 받는 기능이므로 상속 기능에 부가적으로 우리가 처리할 기능을 넣으면 됩니다.

```python
def create(self, request, *args, **kwargs):
    ...
```

<br>

먼저 create 함수를 선언하면 위와 같은 인자를 받게 됩니다.
이 때, request.data 안에 사용자가 입력해서 저장하는 값이 들어있게 됩니다.
이것을 dictionary 형태로 받습니다.

```python
data = request.data.dict()
```

<br>

만약 위도와 경도 2쌍을 받는 다면 dictionary에서 다음과 같이 뽑아낼 수 있습니다.

```python
lat1, long1 = float(data["latitude1"]), float(data["longitude1"])
lat2, long2 = float(data["latitude2"]), float(data["longitude2"])
```

<br>

이제 예를 들면 distance 함수를 통하여 거리를 구한다고 가정합시다.
거리를 구한 다음에 distance field에 값을 저장합니다. 
여기는 그냥 예제이므로 구체적인 것은 이해할 필요가 없습니다.

```python
dist = distance(lat1, long1, lat2, long2)
data.update({"distance" : dist})
```

<br>

## 여기 부분이 중요합니다.

그런데 이렇게 하면 수정이 안됩니다.
왜냐하면 기본적으로 request는 수정이 불가능 하도록 되어있습니다. 데이터의 안정성 때문에 수정이 불가하도록 해놓았는데요.
그러면 저희는 의도적으로 수정을 해야 하기 때문에 다음과 같이 명령어를 입력해 줍니다.

```python
request.POST._mutable = True
```

<br>

이렇게 `._mutable = True`로 입력하면 request가 부분적으로 수정 가능해집니다.
예를 들어 다음과 같이 수정하면 됩니다.

```python
request.POST['distance'] = dist
``` 

이제 마지막으로 부모 클래스로 내용들을 return해 주면 됩니다.

총 정리하면 예제 코드는 다음과 같습니다.

```python
def create(self, request, *args, **kwargs):
    data = request.data.dict()
    lat1, long1 = float(data["latitude1"]), float(data["longitude1"])
    lat2, long2 = float(data["latitude2"]), float(data["longitude2"])
    
    dist = distance(lat1, long1, lat2, long2)
    
    # dictionary를 단순히 수정하는 것은 위에서 설명한 것과 같이 수정이 안됩니다.
    # data.update({"distance" : dist})
    
    request.POST._mutable = True
    request.POST['distance'] = dist

    return super().create(request, *args, **kwargs)
```

<br>

도움이 되셨다면 광고 클릭이 저에게 또한 큰 도움이 됩니다. 꾸벅.