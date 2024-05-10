---
layout: post
title: S3 storage를 위한 boto3 사용법 정리
date: 2021-05-19 13:46:00
img: python/etc/s3_storage_for_boto3/0.png
categories: [python-etc] 
tags: [python, amazon, aws, s3 storage, boto3, s3 upload download, bot3 사용법, s3 upload, s3 download] # add tag
---

<br>

- `S3 storage`를 python을 이용하여 사용하기 위해서는 `boto3`를 사용하면 쉽게 사용할 수 있습니다. document는 아래 내용을 참조하시기 바랍니다.
    - `boto3` document : [https://boto3.amazonaws.com/v1/documentation/api/latest/index.html](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

<br>

- 현재 S3 storage에서 파일 및 폴더를 업/다운로드 하고 싶으시면 아래 깃의 코드를 사용하기를 추천 드립니다. 제가 작성하였고 저 또한 즐겨 사용하고 있습니다. 자세한 사용 방법등은 아래 링크에서 확인할 수 있으며 문의 사항 있으면 댓글 부탁 드립니다.
    - 링크 : [https://github.com/gaussian37/S3UpDownLoader](https://github.com/gaussian37/S3UpDownLoader)

<br>

## **목차**

<br>

- ### [S3 storage에서 폴더 다운 받기](#s3-storage에서-폴더-다운-받기-1)
- ### [S3 storage에 데이터 업로드](#s3-storage에-데이터-업로드-1)
- ### [S3 storage에 폴더 업로드](#s3-storage에-폴더-업로드-1)
- ### [S3 storage에 데이터가 있는 지 확인](#s3-storage에-데이터가-있는-지-확인-1)
- ### [multipart 업/다운로드를 이용한 동시성 작업 (concurrency)](#multipart-업다운로드를-이용한-동시성-작업-concurrency-1)
- ### [S3 storage 파일 크기 확인](#s3-storage-파일-크기-확인-1)
- ### [원하는 위치의 폴더 경로 읽기](#원하는-위치의-폴더-경로-읽기-1)
- ### [S3 storage의 json 파일 읽기](#s3-storage의-json-파일-읽기-1)

<br>

## **S3 storage에서 폴더 다운 받기**

<br>

```python
import boto3
import os 

# set aws credentials 
s3r = boto3.resource(
    's3', 
    aws_access_key_id='xxxxxxxxxxxxxxxxx',
    aws_secret_access_key='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
)
bucket = s3r.Bucket('bucket_name')

# downloading folder 
prefix = 'dirname'
for object in bucket.objects.filter(Prefix = 'dirname'):
    if object.key == prefix:
        os.makedirs(os.path.dirname(object.key), exist_ok=True)
        continue;
    bucket.download_file(object.key, object.key)
```
<br>

## **S3 storage에 데이터 업로드**

<br>

```python
import boto3
s3 = boto3.resource('s3')
s3.meta.client.upload_file('/tmp/hello.txt', 'mybucket', 'hello.txt')
```

<br>

## **S3 storage에 폴더 업로드**

<br>

```python
def uploadDirectory(path,bucketname):
        for root,dirs,files in os.walk(path):
            for file in files:
                s3C.upload_file(os.path.join(root,file),bucketname,file)
```

<br>

## **S3 storage에 데이터가 있는 지 확인**

<br>

- 아래 코드를 이용하여 s3 storage에 파일 존재 유무를 확인하는 방법이 가장 간단한 방법 중 하나입니다.
- `path`에 해당하는 부분에 경로 이름 또는 파일 이름을 입력하였을 경우에 일치하면 True를 반환합니다.

```python
def prefix_exits(bucket, path):
    s3_client = boto3.client('s3')
    res = s3_client.list_objects_v2(Bucket=bucket, Prefix=path, MaxKeys=1)
    return 'Contents' in res
```

<br>

## **multipart 업/다운로드를 이용한 동시성 작업 (concurrency)**

<br>

```python
import boto3
from boto3.s3.transfer import TransferConfig

# Set the desired multipart threshold value (5GB) with 5 max concurrency
GB = 1024 ** 3
config = TransferConfig(multipart_threshold=5*GB, max_concurrency=5)

# Perform the transfer
s3 = boto3.client('s3')
s3.upload_file('FILE_NAME', 'BUCKET_NAME', 'OBJECT_NAME', Config=config)
```

<br>

## **S3 storage 파일 크기 확인**

<br>

```python
import boto3

# Create an S3 client
s3_client = boto3.client(
    's3',
    # + options of access_key, secret_key, ...
)

# Specify your bucket name
bucket_name = 'your-bucket-name'

# Specify the object path
object_path = 'path/to/your/object'

# Get object metadata
response = s3_client.head_object(Bucket=bucket_name, Key=object_path)

# Extract the file size
file_size = response['ContentLength']

print(f"The file size is {file_size} bytes.")
```

<br>

## **원하는 위치의 폴더 경로 읽기**

```python
import boto3

# Initialize S3 client
s3_client = boto3.client(
    's3',
    # + options of access_key, secret_key, ...
)

# Specify the S3 bucket and the "directory" path (prefix)
bucket_name = 'your-bucket-name'
prefixes = [
    'path/to/your/directory1/',  # Make sure to include trailing slash
    'path/to/your/directory2/',  # Make sure to include trailing slash
    ...
] 

# Initialize paginator
paginator = s3_client.get_paginator('list_objects_v2')

# Create pagination object
page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

# Iterate through pages and list all objects
all_objects = []
for prefix in prefixes:
    for page in page_iterator:
        if 'Contents' in page:
            all_objects.extend([obj['Key'] for obj in page['Contents'] if obj['Key'] != prefix])

# Print all objects
print(f"Contents of {prefix} in {bucket_name}:")
for obj in all_objects:
    print(obj)
```

<br>

## **S3 storage의 json 파일 읽기**

<br>

```python
import boto3
import json

# Initialize S3 client
s3_client = boto3.client(
    's3',
    # + options of access_key, secret_key, ...
)

# Specify the bucket and the JSON file path
bucket_name = 'your-bucket-name'
json_file_key = 'path/to/your/file.json'

# Get the JSON file object from S3
s3_object = s3_client.get_object(Bucket=bucket_name, Key=json_file_key)

# Read the contents of the JSON file
json_content = s3_object['Body'].read().decode('utf-8')

# Load the JSON content into a Python dictionary
json_data = json.loads(json_content)

# Print or manipulate the JSON data
print(json_data)
```

<br>