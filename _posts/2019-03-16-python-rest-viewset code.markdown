---
layout: post
title: DRF ViewSet에서 사용할 코드 모음
date: 2019-03-16 18:43:00
img: python/rest/rest.png
categories: [python-rest] 
tags: [python, django, rest, viewset] # add tag
---

+ 이번 포스트는 ViewSet에서 코드 작성 시 종종 사용이 필요한 코드들을 편의상 모아놓은 포스트 입니다.

### VewSet 기본 템플릿

```python
class SampleViewSet(viewsets.ModelViewSet):
    """
    Sample REST API 기능 제공
    """
    
    authentication_classes = []
    serializer_class = SampleSerializer
    queryset = Sample.objects.all()
    ...
```
<br>

+ 기본적으로 `ViewSet`을 작성할 때에는 `vewsets.ModelViewSet`을 상속 받습니다.
+ 테스트 용도로 ViewSet을 만들더라도 기본적으로 위에서 선언된 3가지 속성 정도는 선언해야 합니다.
+ `authetication_classes` : ViewSet을 접근하기 위한 권한입니다. 비워두면 모든 사용자가 ViewSet에 접근 가능합니다.
+ `serializer_class` : ViewSet에 연관된 Serializer를 지정해줍니다.
+ `queryset` : 기본적으로 데이터를 가져올 모델과 모델에서 어떤 데이터를 가져올 지 입력해 줍니다.

<br><br>

### CRUD 함수들의 기본 템플릿

```python
def list(self, request, *args, **kwargs):    
    '''
    list API에 대한 설명~
    
    ---
    입출력 설명
    '''
    ...
    return super().list(request, *args, **kwargs)
```

<br>

+ CRUD 함수인 list/retrieve/create/update/partial_update/destory는 기본적으로 인자를 위의 예제 처럼 받습니다.
    + self, request, *args, **kwargs
+ Return 또한 위의 예제처럼 적어주면 됩니다.
    + ```python return super().list(request, *args, **kwargs) ```
    + 여기서 list 자리에 list/retrieve/create/update/partial_update/destory를 입력해 주면 됩니다.
    
<br><br>

### API를 통해 입력 받은 파라미터 다루기

+ API 호출 시 같이 전달 받은 파라미터를 다루려면 입력 받은 파라미터를 변수로 저장해야 합니다.
+ 입력 받은 파라미터들은 `request`를 통해서 접근할 수 있습니다.

```python
# foodCategory 파라미터, 조회 결과 없을 시 None 리턴
foodCategory = request.GET.get("foodCategory", None)
```

<br>

+ 위 예는 API 호출 시 **foodCategory=고기**로 입력이된 값일 경우라고 생각하면,
    + foodCategory 변수에 "고기" 라고 저장됩니다.
    + 만약 파라미터가 입력되지 않은 경우에는 None으로 저장됩니다.

+ 변수로 파라미터를 할당하면 다음과 같이 ORM으로 모델을 조회하여 원하는 값들을 리턴할 수 있습니다.

```python
# foodCategory와 station 그리고 restaurantName을 모두 받았을 경우 :
if foodCategory is not None:
    self.queryset = self.queryset.filter(foodCategory=foodCategory)
                                                 
```    

<br>

+ 또는 입력 받은 파라미터를 `dictionary`로 먼저 할당 받아서 dictionary에 접근하여 사용할 수 있습니다.

```python
# request.data를 통하여 사용자가 입력한 값을 불러 옵니다.
data = request.data.dict()
# 사용자가 입력한 위도/경도/역 정보를 가져 옵니다.
lat, long, station = float(data["latitude"]), float(data["longitude"]), data["station"]
# 위도/경도/역 정보를 이용하여 식당과 역 사이의 거리를 구합니다.
distFromStation = dist(lat, long, station)

# request.data를 변경 가능하도록 만듭니다.
request.POST._mutable = True
# request.data에 식당과 역까지의 거리 입력합니다.
request.POST['distFromStation'] = distFromStation
```

<br><br>

### 입력 받은 파라미터 수정하기

+ 만약 사용자가 호출한 API의 파라미터값의 일부를 수정해서 저장 또는 사용하고 싶으면 다음과 같이 코드를 작성합니다.

```python
# request.data를 통하여 사용자가 입력한 값을 불러 옵니다.
data = request.data.dict()
# 사용자가 입력한 위도/경도/역 정보를 가져 옵니다.
lat, long, station = float(data["latitude"]), float(data["longitude"]), data["station"]
# 위도/경도/역 정보를 이용하여 식당과 역 사이의 거리를 구합니다.
distFromStation = dist(lat, long, station)

# request.data를 변경 가능하도록 만듭니다.
request.POST._mutable = True
# request.data에 식당과 역까지의 거리 입력합니다.
request.POST['distFromStation'] = distFromStation
```

<br>

+ 파라미터값을 변경하려면 ``` request.POST._mutable = True```를 입력한 다음에 직접 변경할 값을 할당합니다.

<br><br>

### 입력 받은 PK 값에 접근하기

+ 입력 받은 Primary Key 값을 가져와서 모델 인스턴스에 접근합니다.

```python
pk = self.kwargs['pk']
# pk에 해당하는 object를 가져옴
qs = Restaurant.objects.get(id=pk)

```

<br><br>