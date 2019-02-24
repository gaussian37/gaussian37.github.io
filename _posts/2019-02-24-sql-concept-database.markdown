---
layout: post
title: 데이터베이스
date: 2019-02-24 00:00:00
img: dl/concept/autoencoder1/autoencoder.png
categories: [sql-concept] 
tags: [sql, 데이터베이스, database] # add tag
---

+ 출처 : [SQL 첫걸음](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=69025381)

+ 데이터베이스는 특정 데이터를 확인하고 싶을 때 간단하게 찾아낼 수 있도록 정리된 데이터 형식 입니다.
+ 데이터베이스 내의 데이터는 영구적으로 보존되어야 합니다.
    + 따라서 데이터베이스의 데이터는 하드디스크나 플래쉬메모리등 비휘발성 저장장치에 저장합니다.

<br>

+ 데이터베이스는 데이터 센어, 개인용 컴퓨터 또는 휴대용 기기에도 내장되어 있습니다.
+ DB(DataBase) : 저장 장치 내에 정리되어 저장된 데이터의 집합
+ DBMS(DataBase Management System) : DB를 효율적으로 관리하는 소프트웨어

<br>

+ DBMS를 사용하면 여러가지 장점이 있습니다.
    + 생산성이 좋아집니다.
        + CRUD와 같은 작업을 할 때 DBMS를 사용하면 시스템 기본 기능 구현이 필요없이 CRUD를 만들어 낼 수 있습니다.
    + 기능성이 좋아집니다.
        + DBMS는 DB를 다루는 기능을 많이 제공합니다. 복수의 유저의 요청에 대응하거나, 대용량의 데이터를 저장하고 검색하는 기능도 제공합니다.
    + 신뢰성이 좋아집니다.
        + 대규모 DB는 많은 요청에 대응할 수 있도록 만들어져 있습니다.
        + 하드웨어를 여러 대로 구성하여 신뢰성을 높이기도 하고 소프트웨어를 통해 확장성(Scalability)과 부하 분산(Load balancing)을 하기도 합니다.
 
<br>
    
+ 데이터베이스를 조작하려면 `SQL`를 사용하면 됩니다.
+ 데이터베이스에도 여러가지가 있지만 그중 관계형 DBMS(RDMBS)를 다룰 때 SQL을 사용합니다.

<br>

+ SQL은 다음과 같이 3가지로 나뉠 수 있습니다.
    + `DML` : Data Manipulation Language
        + 데이터베이스에 새롭게 데이터를 추가하거나 삭제하거나 내용을 갱신하는 등 데이터를 조작할 때 사용
    + `DDL` : Data Definition Language
        + 데이터를 정의하는 명령어. 데이터베이스는 데이터 베이스 객체라는 데이터 그릇을 이용하여 데이터를 관리하는데, 이 같은 객체를 만들거나 삭제하는 명령어
    + `DCL` : Data Control Language
        + 데이터를 제어하는 명령어. DCL에는 트랜잭션을 제어하는 명령과 데이터 접근권한을 제어하는 명령


