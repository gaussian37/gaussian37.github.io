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

## **S3 storage에서 폴더 다운 받기**

<br>

```python
import boto3
import os 

# set aws credentials 
s3r = boto3.resource('s3', aws_access_key_id='xxxxxxxxxxxxxxxxx',
    aws_secret_access_key='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
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

## **s3 storage에 폴더 업로드**

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

```python
import boto3
from botocore.errorfactory import ClientError
def path_exists(path, bucket_name):
    """Check to see if an object exists on S3"""
    s3 = boto3.resource('s3')
    try:
        s3.ObjectSummary(bucket_name=bucket_name, key=path).load()
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise e
    return True

path_exists('path/to/file.html')
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