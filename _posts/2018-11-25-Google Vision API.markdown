---
layout: post
title: Google Vision API 사용 방법  
date: 2018-11-25 00:00:00
img: vision/etc/gcp/gcp.PNG
categories: [vision-etc] 
tags: [vision, google vision api] # add tag
---

윈도우에서 Google Vision API를 사용하는 방법에 대하여 알아보도록 하겠습니다.

Google Vision API를 사용하려면 2가지 방법이 있습니다.
+ Cloud에 데이터를 저장한 후 Cloud 기반으로 API를 사용하는 방법
+ Local에 있는 이미지 파일을 API를 사용합니다.

여기서 다루어 볼 예제는 Local에 있는 이미지 파일에 API를 사용하는 방법을 알아보겠습니다.
당연히 Local에 있는 파일을 이용을 훨씬 더 많이 하겠죠?

전체적인 순서는 다음과 같습니다.

1. API 키 발급 받기
2. GoogleCloudSDK 설치하기
3. 환경 설정
4. 사용하기

<br>

### 1. API 키 발급 받기

+ https://console.cloud.google.com 에 접속 합니다.
+ 사용자 인증 정보 >> 새 프로젝트를 클릭하여 **프로젝트를 생성** 합니다.

![1](../assets/img/vision/etc/gcp/1.PNG)

<br>

새 프로젝트를 생성하면 위와 같은 화면이 생깁니다. 사용자 인증 정보를 클릭 합니다.

![2](../assets/img/vision/etc/gcp/2.PNG)

<br>

+ 이 때, API 키 / OAuth 클라이언트 ID / 서비스 계정 키 중에 `서비스 계정 키`를 클릭합니다.
+ 서비스 계정키를 클릭한 다음, 키 유형으로는 JSON을 선택합시다. JSON이 표준화 되어 사용되다 보니, JSON을 받아서 처리하는 라이브러리도 많고 사용하기가 편합니다.
+ 이 때 생성되는 서비스 계정 키 ID 는 비밀로 간직해야 합니다. 이 ID로 인증을 하여 API를 사용하고 요금이 청구 됩니다.
+ json파일 다운 받기가 실행되면 json 파일을 다운 받습니다. 실질적으로 사용할 떄 이 json파일을 이용하여 인증을 한 후 사용하게 됩니다.

<br>

### 2. GoogleCloudSDK 설치하기

+ https://cloud.google.com/sdk/docs/#windows 여기에 들어가서 GoogleCloudSDKInstaller.exe 파일을 받아 설치 합니다.
+ 설치를 완료하면 Google Could SDK Shell, Google Cloud SDK Power Shell 등이 생성됩니다.
    + 일반 윈도우 command가 아닌 설치 후 생성된 **Google Could SDK Shell을 이용**해야 합니다.
    
### 3. 환경 설정

+ Google Cloud SDK Shell을 실행 합니다.
+ 이 Shell에서는 gcloud 명령어를 사용할 수 있습니다. 먼저 `gcloud init`을 입력하여 초기화를 해줍니다.
+ 다음으로 Vision API를 설치 합니다. 아래 명령어를 수행합니다.
    + pip install --upgrade google-cloud-vision
+ 조금전 저장한 `.json` 파일을 GOOGLE_APPLICATION_CREDENTIALS 에 연동 시켜 줍니다.
    + set GOOGLE_APPLICATION_CREDENTIALS=".json파일 경로 입력"
    + 세팅을 완료 한 후 Shell을 껐다가 다시 한번 켜봅니다.
    
### 4. 사용하기

... 작성 중 ...
    
    
    


    
    




 

 