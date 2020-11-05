---
layout: post
title: Application Layer
date: 2020-10-02 00:00:00
img: etc/network/0.png
categories: [etc-network] 
tags: [computer network, network, application layer] # add tag
---

<br>

[Computer Network 관련 글 목차](https://gaussian37.github.io/etc-network-table/)

<br>

- 이번 글에서는 OSI 7 계층 중에서 가장 상위에 있는 Application Layer에 대하여 다루어 보겠습니다.

<br>

## **목차**

<br>

- ### [Principles of network applications](#principles-of-network-applications-1)
- ### [Web and HTTP](#web-and-http-1)
- ### [electronic mail (SMTP, POP3, IMAP)](#electronic-mail-smtp-pop3-imap-1)
- ### [DNS](#dns-1)
- ### [P2P application](#p2p-application-1)
- ### [video streaming and content distribution networks](#video-streaming-and-content-distribution-networks-1)
- ### [socket](#socket-1)

<br>

## **Principles of network applications**

<br>
<center><img src="../assets/img/etc/network/application_layer/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 먼저 이 글에서 다룰 전반적인 내용은 application layer와 transport layer 간의 서비스 모델입니다. application layer에서 사용하는 대표적인 서비스 모델로는 `client-server paradigm`과 `peer-to-peer paradigm`이 있습니다.
- 그리고 `content distribution network`에 대해서도 다루어 보려고 합니다.

<br>
<center><img src="../assets/img/etc/network/application_layer/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- application 에서 사용하는 다양한 프로토콜에 대해서도 다루어 보려고 합니다. 대표적으로 HTTP, FTP, SMTP, POP3, IMAP, DNS등이 있습니다. 

<br>
<center><img src="../assets/img/etc/network/application_layer/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- network application은 application 개발자에 의해 만들어지고 각각의 application은 각 네트워크의 종단에서 사용되어 집니다.
- 이 때 application 개발자들이 모든 네트워크를 신경써서 개발을 하려면 상당히 골치아파지는데, 개발의 효율성을 위해서 application 개발자들은 application layer만 신경을 쓰면 된다는 것이 핵심입니다.
- 왜냐하면 OSI 계층의 컨셉 상 각 layer에서는 각 layer와 인접한 layer에만 신경쓰면 되기 때문입니다.
- 참고로 컴퓨터, 스마트폰과 같이 종단에 있는 시스템은 위 그림과 같이 physical, data link, network, transport, application으로 이어지는 계층을 가지고 있는 반면, 중간 중간의 라우터 들은 physical, data link, network까지만 가지고 있습니다.

<br>
<center><img src="../assets/img/etc/network/application_layer/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 대표적인 application architecture는 `client-server`와 `P2P(peer-to-peer)` 방식입니다.
- client-server는 server가 고정되어 있고 client들이 접근해서 사용하는 방식입니다. 예를 들어 게임이나 서버의 파일을 다운 받는 것 등이 해당됩니다.
- 반면, P2P에서는 server가 고정되어 있지 않습니다.

<br>
<center><img src="../assets/img/etc/network/application_layer/6.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 먼저 client-server 구조에 대하여 다루어 보겠습니다. cilient-server 구조는 예전부터 현재까지 가장 널리 사용하고 있는 구조 입니다.
- server는 항상 host로써 켜져 있어야 하고 고정된 IP 주소를 가집니다. 대표적인 예로 데이터 센터가 있습니다.
- 반면 client는 server와 통신을 하며 상시 연결되지 않고 사용 시에만 서버와 통신을 합니다. 간헐적으로 사용하기 때문에 동적으로 IP 주소를 할당 받습니다.
- server 연결 없이 client 끼리는 직접 통신할 수는 없습니다.

<br>
<center><img src="../assets/img/etc/network/application_layer/7.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 반면 P2P에서는 상시 켜져있는 server는 없습니다. 대신에 client와 client가 직접 통신하는 방식입니다. server-client에서는 상시 client와 통신을 해야 하는 server가 있어야 하기 때문에 그 server를 마련하기 위한 비용이 존재하지만 P2P에서는 그 비용을 고려할 필요는 없습니다. 
- 하지만 상시 통신이 가능한 server가 없어지면서 상당히 복잡해집니다. 서비스를 제공하는 client가 중간에 사라질 수도 있고 client의 IP가 유동적이기 때문에 이를 관리하기도 복잡해집니다.

<br>
<center><img src="../assets/img/etc/network/application_layer/8.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- Application layer에서 가지는 개념으로 `Process`가 있습니다. 
- Operating System에서 말하는 프로세는 어떤 프로그램을 실행시켰을 때, 프로그램을 수행하는 것을 말합니다. 한 개의 프로그램에 여러 개의 process가 할당될 수 있습니다.
- Application layer에서도 비슷합니다. **host에서 수행 중인 프로그램**을 process 라고 합니다. 여기서도 한 개의 프로그램에 여러 process가 할당될 수 있습니다. 이 때, 각 process 간 통신을 하는 것을 **inter-process communication**이라고 합니다.
- 반면 서로 다른 host에 있는 process 끼리 통신이 필요한 경우도 있습니다. 이 때는 Application layer에서 message 단위의 정보를 서로 주고 받으면서 통신하게 됩니다.
- client - server 구조에서 client에서 client process가 동작하고 server에서 server process가 동작합니다. 특히 server process는 항상 동작하고 있습니다. 왜냐하면 client process가 통신을 하게 되면 그에 알맞는 서비스를 제공해야 하기 때문입니다.
- 반면 P2P에서는 한 개의 host에서 server, client process가 동시에 작동할 수 있습니다.

<br>
<center><img src="../assets/img/etc/network/application_layer/9.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 앞에서 설명한 바와 같이 서로 다른 host에 있는 process 끼리 통신이 필요한 경우가 있습니다.
- 서로 다른 host 간 통신을 하기 위하여 process는 message를 주고 받습니다. 이 때 사용되는 것이 socket입니다.
- 한 host에서 다른 host로 message를 보내려면 application layer → transport layer를 거쳐야 합니다. application layer에서는 transport layer를 관여하지 않기 때문에 단순히 socket을 통하여 message를 전달하기만 하면 그 이후의 작업은 application layer 이 외의 layer에서 처리하게 됩니다.
- 따라서 socket은 마치 문에 비유되곤 합니다. message를 보내기 위해서 socket이라는 문을 통해 transport layer로 전달되기 때문입니다.

<br>
<center><img src="../assets/img/etc/network/application_layer/10.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 서로 다른 host에서 process 간 메시지를 주고 받을 때, 어디로 메시지를 전달해야 하는 지 정보가 필요합니다. 바로 `IP주소`와 `Port 넘버`로 구성된 `identifier`입니다.
- 먼저 어떤 host로 전달해야 하는 지에 대한 정보는 **32bit의 IP 주소**를 통해서 확인할 수 있습니다.
- 또한 각 host에서 어떤 process에 전달해야 하는 지 알아야 합니다. 앞에서 설명하였듯이 host에 process는 여러개 존재할 수 있으므로 IP 주소 이외에 추가적인 정보가 더 필요합니다. 이 정보가 **Port 넘버**입니다.
- 따라서 **IP 주소**와 **Port 넘버**를 포함하는 `identifier`를 이용하여 어떤 host의 어떤 process와 통신할 수 있는 지 알 수 있습니다.
- 일부 Port 넘버는 표준으로 정하여 사용하는 것도 있습니다. 예를 들어 HTTP (80), mail server (25) 등이 있습니다.

<br>
<center><img src="../assets/img/etc/network/application_layer/11.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- Application layer에서 정의되는 프로토콜에 대하여 알아보도록 하겠습니다.
- 먼저 서로 주고 받는 메시지의 **타입**이 정의 됩니다. 예를 들어 request 메시지와 response 메시지등이 있습니다.
- 각 메시지들은 컴퓨터가 이해하기 위하여 bit로 되어 있습니다. 이 bit를 의미 있게 만들기 위하여 **syntax**와 **semantics**가 정의 됩니다. **syntax**는 메시지의 bit 영역을 나누는 기준이 되고 **semantics**는 각 영역에서 bit 값이 어떤 의미를 가지는 지에 대한 프로토콜이 됩니다.
- 또한 각종 규칙과 open 프로토콜 등이 application layer에서 정의될 수 있습니다.

<br>
<center><img src="../assets/img/etc/network/application_layer/12.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- Application layer에서 직접적으로 주고 받는 layer는 transport layer입니다. 그러면 transport layer에서 application layer에 원하는 요구 사항들이 있습니다. 대표적으로 `data integrity`, `timing`, `throughput` 등이 있습니다.
- 기본적으로 transport layer에서는 결함 없는 데이터(**data integrity**)를 받기를 원합니다. 결함의 정도는 앱에 따라 다르며 어떤 앱은 100% 결함없는 데이터를 필요로 하는 반면 어떤 앱은 일정 부분 결함을 인정하기도 합니다. 대신 빠른 전송 속도를 원하는 것이지요.
- **timing**과 관련된 내용으로는 delay를 줄여서 효율적으로 데이터를 전달하기를 원합니다.
- **throughput**과 관련된 내용으로는 어떤 시간 안에 최소한 이 정도 일은 처리를 해야 한다는 일 처리 양과 관련된 요구 사항이 있을 수 있습니다.

<br>
<center><img src="../assets/img/etc/network/application_layer/13.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 예를 들어 위 테이블과 같이 application의 종류에 따라서 **data loss**, **throughput**, **timing**에 얼만큼 민감한지 분류할 수 있습니다.
- `elastic`은 어느 정도 throughput 양이 왔다 갔다 할 수 있다는 유연함을 나타냅니다.

<br>
<center><img src="../assets/img/etc/network/application_layer/14.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- transport layer에서 사용하는 대표적인 프로토콜은 TCP와 UDP가 있습니다. 이 내용은 다음 글에서 다룰 예정이지만 간략하게 알아보겠습니다.
- TCP의 가장 큰 특징은 data integrity 입니다. 즉, 데이터의 결함이 없도록 보내는 것에 목적이 있습니다. 따라서 결함이 있는 것을 확인하면 재전송 하는 절차를 가집니다. 
- 또한 TCP에서는 flow control을 합니다. 이는 sender에서 데이터를 계속 보내지만 receiver에서 받아서 처리할 용량이 안되면 보낸 데이터를 제대로 처리할 수 없어서 결함이 생기므로 sender 쪽에 이 문제를 알려줘서 sender에서 데이터 전송을 조절하도록 합니다.
- TCP에서 하는 congestion control은 네트워크 자체에 과부하가 걸리는 경우에 해당합니다. 이 경우 또한 데이터가 제대로 전달이 안될 수 있으므로 sender 쪽에 알려줘서 데이터 전송을 조절하도록 합니다.
- 이 모든 것이 **데이터가 결함 없이 전달되도록 하는 것**에 목적이 있습니다.

<br>

- 반면 UDP에서는 데이터 무결성에 관심이 없습니다. 그저 빨리 빨리 보내는 것에 목적이 있습니다. 즉, 중간 중간에 데이터 손실이 생기더라도 빨리 빨리 데이터를 전달해서 실시간 성능을 확보해야 하는 어플리케이션이 유리합니다.

<br>

- TCP, UDP의 **does not provide** 항목을 보면 timing과 throughput에 대한 최소한의 어떤 요구 사항은 보장할 순 없습니다. 즉, 네트워크 상에서 발생할 수 있는 최소한의 한계는 TCP, UDP 모두 가지고 있습니다.

<br>
<center><img src="../assets/img/etc/network/application_layer/15.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 앞의 내용을 참조하여 어떤 어플리케이션들은 반드시 TCP를 써야 하고 어떤 어플리케이션은 UDP를 사용해도 되는 지 살펴 보시기 바랍니다.


<br>

## **Web and HTTP**

<br>


<br>

## **electronic mail (SMTP, POP3, IMAP)**

<br>

<br>

## **DNS**

<br>


<br>

## **P2P application**

<br>

<br>

## **video streaming and content distribution networks**

<br>


<br>
<center><img src="../assets/img/etc/network/application_layer/77.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- DASH라는 서비스는 multimedia streaming 서비스를 할 때, 네트워크 상태가 나쁘면 저화질로 streaming을 하다가 네트워크 상태가 좋아지면 고화질로 streaming을 하기 위하여 개발되었습니다. 이는 현재 네트워크 상태에서 가능한한 고화질로 streaming을 하기 위함입니다.
- DASH를 위하여 server 측면에서는 비디오 파일을 여러개의 chunk로 나눕니다. 그리고 각각의 chunk를 저장할 때, 다른 rate로 인코딩을 하여 저장을 합니다. 즉, 같은 chunk라도 다른 rate로 인코딩이된 비디오를 가집니다.
- 이 때, 서로 다른 chunk와 인코딩 rate가 다른 같은 chunk에 대하여 URL을 제공하는 manifest file이라는 것을 가지게 됩니다.
- 이 파일은 streaming 서비스를 받는 소프트웨어 (client) 에서 사용하게 됩니다.
- client에서는 주기적으로 server와 client 간의 bandwidth를 측정하고 bandwidth가 허용하는 최대한의 rate로 chunk를 받습니다. 예를 들어 bandwidth의 상태가 안좋을 때에는 저화질로 인코딩 된 chunk를 받고 상태가 좋을 때에는 고화질로 인코딩 된 chunk를 받습니다. 이 때 manifest file을 보고 그 상황에 맞는 chunk의 URL을 사용하게 됩니다.

<br>

## **socket**

<br>



<br>

[Computer Network 관련 글 목차](https://gaussian37.github.io/etc-network-table)

<br>
