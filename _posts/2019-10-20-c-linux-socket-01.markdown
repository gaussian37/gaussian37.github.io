---
layout: post
title: 네트워크 프로그래밍과 소켓의 이해
date: 2019-10-20 00:00:00
img: c/linux/socket/socket.png
categories: [c-linux] 
tags: [소켓 프로그래밍] # add tag
---

<br>

- 출처: TCP/IP 소켓 프로그래밍

<br>

## **목차**

<br>

- ### 1. 네트워크 프로그래밍과 소켓의 이해
- ### 2. 간단한 서버 코드 (연결을 받는 소켓) 살펴보기
- ### 3. 간단한 클라이언트 코드 (연결을 요청하는 소켓) 살펴보기
- ### 4. 서버 코드와 클라이언트 코드 작동
- ### 4. 리눅스 기반 파일 조작
- ### 5. 윈도우 기반으로 구현
- ### 6. 윈도우 기반 소켓 관련 함수와 예제

<br>

## **1. 네트워크 프로그래밍과 소켓의 이해**

<br>

- 네트워크 프로그래밍의 정의를 먼저 살펴보면 네트워크로 연결된 둘 이상의 `컴퓨터 사이에서의 데이터 송수신` 프로그램의 작성을 의미합니다.
- 소켓이라는 것을 기반으로 프로그래밍을 하기 때문에 소켓 프로그래밍이라고도 부르기도 합니다.
- 따라서 네트워크 프로그래밍을 할 때는 운영체제에서 `소켓`이라는 소프트웨어 모듈을 제공해주고 그것을 이용하여 프로그래밍을 합니다. 
- `소켓`을 이용하면 내부적으로 어떻게 통신하는 지 정확하게 알지 못하더라도 컴퓨터 끼리 네트워크 상에서 데이터를 주고 받을 수 있습니다.
- 이번 글에서 다룰 내용은 `소켓`을 이용하여 어떻게 데이터를 주고 받는지 간략하게 살펴보려고 합니다.

<br>

- 먼저 소켓은 `전화기`에 비유해 볼 수 있습니다. 따라서 소켓을 생성한다는 것은 전화기를 한 대 구입한다는 것으로 이해할 수 있습니다.
- 소켓은 `socket` 함수의 호출을 통해서 생성될 수 있습니다. 단, 일반 전화기와의 차이점은 전화를 거는 용도의 소켓과 전화를 수신하는 용도의 소켓 생성 방법에 차이가 있다는 것입니다.

<br>

```cpp
#include <sys/socket.h>

int socket(int domain, int type, int protocol);
```

<br>

- 위 함수를 통하여 소켓을 생성합니다. 성공 시 `파일 디스크립터`를 반환하고 실패 시 -1을 반환합니다.

<br>

- 소켓을 할당한 것을 전화기를 사는 것에 비유해 보았습니다. 전화기를 샀으면 그 전화기에 사용될 전화번호가 필요하겠죠?
- 전화기에 전화번호가 부여되듯이 **소켓에도 주소정보가 할당**됩니다.
- 소켓의 주소정보는 `IP`와 `PORT`번호로 구성이 됩니다.

<br>

```cpp
#include <sys/socket.h>

int bind(int sockfd, struct sockaddr *myaddr, socklen_t addrlen);
```

<br>

- `bind`함수를 통하여 주소를 할당합니다. 성공 시 0을 실패 시 -1을 반환합니다.

<br>

- 다음으로 해야할 작업은 `전화기를 연결`하는 것입니다.
- 연결 요청이 가능한 상태의 소켓은 걸려오는 `전화를 받을 수 있는 상태`에 비유할 수 있습니다.
- 이 때, 전화를 거는 용도의 소켓은 연결 요청이 가능한 상태의 소켓이 될 필요가 없습니다. 따라서 이것은 전화를 받는 용도의 소켓에서만 필요한 상태입니다.

<br>

```cpp
#include <sys/socket.h>

int listen(int sockfd, int backlog);
``` 

<br>

- `listen` 함수를 호출하게 되면 소켓에 할당된 IP와 PORT번호로 연결 요청이 가능한 상태가 됩니다.
- 이 때, 성공 시에는 0을 실패시에는 -1을 받게됩니다.
 
<br>

- 그 다음으로 다룰 함수는 `accept` 함수 입니다. 바로 전에 `listen` 함수를 호출하게 되면 이 함수를 호출한 소켓은 연결이 될 준비가 되어 있는 상태이므로
- 어떤 연결 요청이 들어오면 `accept` 함수를 통하여 전화를 받는 행위를 해주어야 합니다. 
- 두 소켓간 연결이 되면 데이터 송수신이 가능하고 이것은 양방향 송수신이 됩니다. 

<br>

```cpp
#include <sys/socket.h>

int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen);
```

<br>

- `accept` 함수 호출 성공 시, 파일 디스크립터를 반환합니다. 
- 두 소켓 간에 연결이 되면 마치 파일 입출력을 하듯이 프로그램을 하면 네트워크 상에서 데이터를 주고받을 수 있습니다.

<br>

- 먼저 `연결 요청을 허용하는 소켓의 생성과정`을 정리하면, **소켓의 생성(socket) → IP와 PORT번호 할당(bind) → 연결 가능한 상태로 변경(listen) → 연결요청에 대한 수락(accept)** 순서로 이루어집니다.
- 이러한 과정을 거치는 프로그램을 `서버`라고 부르고 있습니다. 
- 일반적으로 `서버`를 보면 연결을 요청하는 클라이언트보다 먼저 실행되어야하고 복잡한 실행 과정을 거치게 됩니다.

<br>

## **2. 간단한 서버 코드 (연결을 받은 소켓) 살펴보기**

<br>

- 아래 코드를 통하여 위해서 설명한 네트워크 프로그래밍의 기본적인 절차를 한번 이해해 보시기 바랍니다.
- 주석을 보면서 이해하시고 여기서는 자세한 함수의 설명을 다 이해하려고 하지 마시고 전체적인 흐름만 살펴보는 것으로 만족해 봅시다!
- 각 함수의 자세한 설명은 코드를 살펴본 뒤 다루겠습니다. 일단 나무를 보기보다 숲을 한번 살펴보자는 것이죠.

<br>

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

void ErrorHandling(char *message);

int main(int argc, char *argv[]){

    int serv_sock;
    int clnt_sock;

    struct sockaddr_in serv_addr;
    struct sockaddr_in clnt_addr;
    socklen_t clnt_addr_size;

    char message[] = "Hello world";

    if(argc != 2){

        printf("Usage : %s <port>\n", argv[0]);
        exit(1);
    }

    // socket을 생성합니다. socket은 운영체제가 관리하고 있습니다.
    // 여기서 socket을 단지 생성하기만 하면 여기서 생성한 socket이 어떤 socket인 지 알 수 없습니다.
    // 따라서 운영체제는 socket 함수 호출을 통하여 생성한 socket에 번호를 부여합니다.
    // 아래 socket 함수에서 반환되는 정수형 값이 바로 그 번호 입니다. 
    // 이 번호를 file descriptor 또는 socket handle 이라고 합니다.
    serv_sock = socket(PF_INET, SOCK_STREAM, 0);
    if(serv_sock == -1){
        ErrorHandling("socket() error");
    }

    // serv_addr 구조체 변수에 IP와 PORT 정보를 저장합니다.
    // 정보를 저장하기 전에 초기화를 해줍니다.
    memset(&serv_addr, 0, sizeof(serv_addr));
    // 아래 3줄을 통하여 IP 주소와 PORT 번호를 할당해줍니다.
    // 상세 내용은 이후에 알아보고 아래 작업을 통하여 IP, PORT가 할당된다는 것만 확인하고 넘어가겠습니다.
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(atoi(argv[1]));

    // 한 프로그램 내에서는 여러개의 socket을 생성할 수 있습니다.
    // 따라서 어떤 socket에 해당하는 IP와 PORT 정보를 할당하기 위해서 bind 함수에서는 socket의 file descriptor를 인자로 넘겨줍니다.
    // 즉, serv_sock에 해당하는 socket에 serv_addr 주소 정보를 할당해 주는 코드입니다.
    if( bind(serv_sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) == -1 ){
        ErrorHandling("bind() error");
    }

    // serv_sock에 해당하는 socket이 연결 가능한 상태가 되도록 listen 함수를 호출합니다.
    if( listen(serv_sock, 5) == -1){
        ErrorHandling("listen() error");
    }

    // accept 함수는 blocking 함수 역할을 합니다. 즉, client의 연락이 올 때 까지 계속 기다리게 됩니다.
    // client의 연락이 오게 되면 client socket의 file descriptor를 반환하고 다음 코드 라인으로 넘어가게 됩니다.
    clnt_addr_size = sizeof(clnt_addr);
    clnt_sock = accept(serv_sock, (struct sockaddr*)&clnt_addr, &clnt_addr_size);
    if(clnt_sock == -1){
        ErrorHandling("accept() error");
    }

    // server와 client의 연결이 되고 나면 server와 client 간의 데이터 송수신이 가능해 지게 됩니다.
    // 아래 write를 통하여 server에서 client 쪽으로 데이터를 보낼 수 있습니다.
    write(clnt_sock, message, sizeof(message));

    // close 함수를 통하여 생성한 socket을 닫아주도록 운영체제 쪽으로 요청할 수 있습니다.
    // 여기서 저희가 생성한 socket은 server의 socket인데 client socket 까지 같이 close 요청을 하고 있습니다.
    // 관련 내용은 이후에 또 자세하게 다루어 보겠습니다. 일단 큰 틀로 보았을 때 이렇게 하면 간단한 네트워크 프로그래밍이 완료됩니다.
    close(clnt_sock);
    close(serv_sock);
    return 0;

}

void ErrorHandling(char *message){
    fputs(message, stderr);
    fputc('\n', stderr);
    exit(1);
}
```

<br>

## **3. 간단한 클라이언트 코드 (연결을 요청하는 소켓) 살펴보기**

<br>

- 연결을 요청하는 소켓의 구현에 대하여 다루어 보도록 하겠습니다.
- 앞에서 다룬 서버 코드와 비교하면 상당히 단순한 것이 `connect` 함수를 이용하여 연결만 하기 때문입니다.
- 




