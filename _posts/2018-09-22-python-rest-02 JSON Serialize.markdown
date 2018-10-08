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
| Specify form field / read from model 	| Warm 	|
| Create Form HTML 	| Create JSON serialization 	|
| Check validation about input data and acquisition 	| Cold 	|




