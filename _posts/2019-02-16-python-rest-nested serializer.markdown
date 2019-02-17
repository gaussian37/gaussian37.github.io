---
layout: post
title: DRF에서 Nested Serializer 사용법
date: 2019-02-16 00:00:00
img: python/rest/rest.png
categories: [python-rest] 
tags: [python, django, REST, drf, serializer, nested serializer] # add tag
---

+ 이번 글에서는 Serializer 안에 또 다른 Serializer가 중첩된 `nested serializer` 형태를 생성하는 방법에 대하여 알아보겠습니다.
+ 다양한 방법의 `nested serializer`가 있으며 상황에 따라 사용방법이 조금씩 다를 수 있습니다.

## Nest serializer 사용 예시 목록

+ 1 : N 관계에서 `1`에 해당하는 model 기준으로 nested serializer 생성 (Reverse relations)
+ 1 : N 관계에서 `N`에 해당하는 model 기준으로 nested serializer 생성

---

###  1 : N 관계에서 1에 해당하는 model 기준으로 nested serializer 생성 (Reverse relations)

+ 참고 자료 : https://www.django-rest-framework.org/api-guide/relations/#reverse-relations
+ 1 : N 관계에서 1에 해당하는 model을 기준으로 nested serializer를 생성하는 방법에 대하여 알아보겠습니다.
    + 이 방법을 djangro restframework에서는 `Reverse relations` 라고 정의하였습니다.
+ 1 : N 관계이므로 1에 해당하는 model의 `Primary Key`를 N에 해당하는 모델에서 `Foreign Key`로 사용하고 있습니다.
+ 아래 예제는 `ModelSerializer`를 기준으로 작성하였습니다.

<br>

+ 아래 코드에서 Album : Track = 1 : N의 관계를 가집니다.
+ reverse relation을 정의하려면 Serializer의 field에 정의를 해야 합니다.
    + AlbumSerializer 클래스 내부에 `tracks = TrackSerializer(many=True, read_only=True)` 이 있습니다.

```python
class TrackSerializer(serializers.ModelSerializer):
    class Meta:
        model = Track
        fields = ('order', 'title', 'duration')

class AlbumSerializer(serializers.ModelSerializer):
    tracks = TrackSerializer(many=True, read_only=True)

    class Meta:
        model = Album
        fields = ('album_name', 'artist', 'tracks')
```

<br>

+ 만약 위 코드 처럼 AlbumSerializer 내부에 `tracks`라는 변수명으로 입력을 받기 위한 조건
    + Track 모델 내부의 album 모델 관련 `Foreign Key`에서 related_name으로 **tracks**를 지정해 주어야 합니다.

```python
class Track(models.Model):
    album = models.ForeignKey(Album, related_name='tracks', on_delete=models.CASCADE)
```

<br>

+ 만약 Foreign Key에서 related name을 지정해 주지 않는 다면 기본 값인 `모델명_set`으로 설정할 수 있습니다.
+ related name을 설정하지 않은 경우 AlbumSerializer에서 따로 Trackserializer()를 통하여 할당 받지 않고 field 명에 track_set으로 사용 가능합니다.

```python
class AlbumSerializer(serializers.ModelSerializer):
    class Meta:
        fields = ('track_set', ...)
``` 

<br>

+ related_name 지정 유무와 상관없이 결과는 동일하고 nested serializer 결과는 다음과 같습니다.
+ Album의 Primary Key에 해당하는 모든 Track 모델의 값을 join 해서 리턴해 줍니다.

```python
    'album_name': 'Sometimes I Wish We Were an Eagle',
    'artist': 'Bill Callahan',
    'tracks': {
        'Track 1: Jim Cain (04:39)',
        'Track 2: Eid Ma Clack Shaw (04:19)',
        'Track 3: The Wind and the Dove (04:34)',
        ...
    }
```

<br><br>

### 1 : N 관계에서 N에 해당하는 model 기준으로 nested serializer 생성

+ 위의 예제에서는 1 : N의 관계에서 1에 해당하는 model의 Serializer에서 nested serializer를 생성하였습니다.
+ 이번에는 1 : N의 관계에서 N에 해당하는 model의 Serializer에서 nested serializer를 만들어 보겠습니다.
+ 이번 예제는 Child와 Parent 입니다. 한명의 Child에게는 2명의 부모님이 있으니 Child : Parent = 1 : N 의 관계 입니다.

```python
class Child(models.Model):
    name = CharField(max_length=20)

class Parent(models.Model):
    name = CharField(max_length=20)
    phone_number = models.ForeignKey(PhoneNumber)
    child = models.ForeignKey(Child)

class ChildSerializer(ModelSerializer):
    class Meta:
        model = Child
        fields = '__all__'


class ParentSerializer(ModelSerializer):
    class Meta:
        model = Parent
        fields = '__all__'

    def to_representation(self, instance):
        response = super().to_representation(instance)
        response['child'] = ChildSerializer(instance.child).data
        return response
```

<br>

+ 여기서 핵심은 `ParentSerializer`에 override 된 `to_representatin`입니다.
    + `response[모델명] = ModelSerializer(instance.ForeingKeyField).data` 로 선언하면 됩니다.
    + 위 예제에서는 response\['child'\] = ChildSerializer(instance.child).data
+ 위 코드와 같이 구성 시 list를 조회하면 아래와 같이 결과를 리턴 받습니다.    

```python
{
    "id" : 4,
    "name" : "parantname",
    "phone_number" : 123-123,
    "child" : {
        "id" : 1,
        "name" : "childname"
    }
    
}
```

<br><br>




