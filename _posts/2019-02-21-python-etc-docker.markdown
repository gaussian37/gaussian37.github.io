---
layout: post
title: Docker 와 Azure Containers for Web App 사용하기
date: 2019-02-21 13:46:00
img: python/etc/docker/docker.png
categories: [python-etc] 
tags: [python, docker, 도커] # add tag
---

+ 출처 : https://www.askcompany.kr/

## 간략히 Docker 살펴보기

### Docker 란?

+ 빠르고 가벼운 가상화 솔루션
+ 애플리케이션과 그 실행환경/OS를 모두 포함한 소프트웨어 패키지
    + Docker Image 
+ 플랫폼에 상관없이 실행될 수 있는 애플리케이션 **컨테이너를 만드는 기술**
+ Docker Image는 Container의 형태로 Docker Engine이 있는 어디에서나 실행 가능
    + 대상 : 로컬 머신(윈도우/맥/리눅스), Azure, AWS, Digital Ocean 등
    + 하나의 Docker Image를 통해 다수의 Container를 생성할 수 있습니다.
+ 생성된 Docker Container는 바로 쓰고 버리는 것 (Immutable Infrastructure 패러다임)
+ Docker Container는 격리되어있어서, 해킹되더라고 Docker Engine이 구동되는 원래의 서버에는 영향을 끼치지 않음

<br>

### Docker만의 특징/유의사항

+ Docker 내에서 어떤 프로세스가 도는 지 명확히 하기 위해서
    + 하나의 Docker 내에서 다양한 프로세스가 구동되는 것을 지양합니다.
        + **한 종류의 프로세스만을 구동하는 것을 지향합니다.**
    + 하나의 Docker 내에서 프로세스를 Background 로 구동하는 것을 지양합니다.
        + **프로세스를 Foreground로 구동하는것을 지향합니다.**
            + nginx 예시 : nginx -g daemon off;
        + **실행 로그도 표준출력(stdout)으로 출력합니다.**
        
<br>

### Container Orchestration

+ 컨테이너 관리 툴의 필요성
    + 컨테이너 자동 배치 및 복제, 컨테이너 그룹에 대한 로드 밸런싱, 컨테이너 장애 복구, 클러스터 외부에 서비스 노출,
    + 컨테니터 추가 또는 제거로 확장 및 축소, 컨테이너 서비스 간의 인터페이스를 통한 연결 및 네트워크 포트 노출 제어
+ 우리가 사용해 볼 `Azure Container for Web App`은 웹 서비스 전용

<br>

### Docker Registry

+ Docker 이미지 저장소를 뜻합니다.
+ 공식 저장소는 Docker Hub : https://hub.docker.com/ (Docker 계의 GitHub)
+ Azure Containers for Web App 에서는 지정 Docker Registry로부터 이미지를 읽어들여, Docker Container를 적재합니다.

<br>

### Dockerfile

+ Docker 이미지를 만들 때, 수행할 명령과 설정들을 시간순으로 기술한 파일
+ 아래는 Dockerfile의 예제입니다.
    + Dockerfile의 첫 글자는 대문자 입니다.

```python
FROM ubuntu:16.04

RUN apt-get update && apt-get install -y python3-pip python3-dev && apt-get clean

RUN mkdir /code

WORKDIR /code

ADD requirements.txt /code/

RUN pip3 install -r requirements.txt

ADD . /code/

EXPOSE 8000
CMD ["python3", "/code/manage.py", "runserver", "0.0.0:8000"]
```

+ `FROM` ubuntu:16.04
    + Docker 이미지는 OS 정보를 가지고 있어야 합니다.
+ `RUN` apt-get update && apt-get install -y python3-pip python3-dev && apt-get clean
    + 각 OS에 맞는(리눅스는 Shell, 윈도우는 명령 프롬프트) 실행창에서 수행할 명령어
    + && 명령어로 연결된 것은 && 명령어 앞의 명령이 수행이 정상적으로 완료되어야 뒤에 명령어가 수행된다는 의미 입니다.
        + 내부적으로 exit code의 return 값이 0이면 성공, 그 이외에는 실패라는 것을 이용하게 됩니다.
+ `WORKDIR` /code/
    + 수행되는 workdir을 설정합니다.
+ `ADD` requirements.txt /code/
    + Docker host측의 파일(requirements.txt)을 Docker image(/code/)로 복사하겠다는 의미 입니다. 
+ `ADD` . /code/
    + Docker host측의 파일을 Docker image로 모두 복사하겠다는 의미 입니다.
+ 
