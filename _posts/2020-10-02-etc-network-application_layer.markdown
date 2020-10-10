---
layout: post
title: Application Layer
date: 2020-10-02 00:00:00
img: etc/network/0.png
categories: [etc-network] 
tags: [computer network, network, table] # add tag
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
- 참고로 컴퓨터, 스마트폰과 같이 종단에 있는 시스템은 위 그림과 같이 physical, data link, network, transpot, application으로 이어지는 계층을 가지고 있는 반면, 중간 중간의 라우터 들은 physical, data link, network까지만 가지고 있습니다.

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

## **socket**

<br>



<br>

[Computer Network 관련 글 목차](https://gaussian37.github.io/etc-network-table)

<br>
