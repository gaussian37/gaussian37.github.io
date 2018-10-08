---
layout: post
title: JSON Serialize
date: 2018-09-22 14:35:00
img: python/rest/rest.png
categories: [python-rest] 
tags: [python, django, rest] # add tag
---

### JSON Serialization through ModelSerializer

In django-rest-framework(DRF), first transform original data to convertible data in JSONRender through `ModelSerializer`.
`Serializer` is similar with Django's `Form`, `ModelSerializer` is Django's `ModelForm`.
In the role of those, `Serializer` is just Form which only proceeds `POST`.

| Django Form/ModelForm 	| Django Serializer/ModelSerializer 	|
|:-------------------------------------------------:	|:---------------------------------:	|
| Specify form field / read from model 	| ← 	|
| Create Form HTML 	| Create JSON serialization 	|
| Check validation about input data and acquisition 	| ← 	|

Define `ModelSerializer`. It's really similar to `ModelForm`.

```python
from rest_framework.serializers import ModelSerializer

# Define ModelSerializer about Post Model
class PostModelSerializer(ModelSerializer):
    class Meta:
        model = Post
        fields = '__all__'
```

<br>

`PostModelSerializer` supports transformation to `dict` type for Post model instance.
Feed Post instance to `PostModelSerializer`.

```python
post = Post.objects.first()  # Post type
serializer = PostModelSerializer(post)
serializer.data 
```

serializer.data type is `ReturnDict`, which inherited from `OrderDict` and get the serializer field additionally through generator.

```python
class ReturnDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        self.serializer = kwargs.pop('serializer')
        super().__init__(*args, **kwargs)
        # ...
```

### 