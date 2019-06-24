---
layout: post
title: Problem Solving 글 목차
date: 9999-01-01 00:00:00
img: interview/ps/ps.png
categories: [interview-ps] 
tags: [ps, c++] # add tag
---

- Problem Solving 문제 리스트 입니다. 
- 1개의 문제라도 깊고 확실히 알 수 있도록 훈련해 보고 있습니다.

### 문제 풀 때 필요한 C++ 관련 내용

- [C++에서 문제 풀 때 좋은 문법 및 템플릿](https://gaussian37.github.io/interview-ps-tip/)
- [C++에서 문제 풀 때 좋은 STL](https://gaussian37.github.io/interview-ps-stl/)

<br>

## 면접에는 나올 수준의 문제 중 풀어볼 만한 문제 코드 및 설명


- `그리디`
    - [가장 큰 수](https://gaussian37.github.io/interview-ps-p42746/)(https://programmers.co.kr/learn/courses/30/lessons/42746)
        - 정렬할 때 새로운 기준을 줘서 정렬하는 문제 : comparator 사용
    - [큰 수 만들기](https://gaussian37.github.io/interview-ps-p42883/)(https://programmers.co.kr/learn/courses/30/lessons/42883)
        - 그리디 방법으로 숫자문자열에서 특정 숫자 k개를 삭제하였을 때 가장 큰 숫자를 만드는 방법을 생각하는 문제
    - [구명보트](https://gaussian37.github.io/interview-ps-p42885/)(https://programmers.co.kr/learn/courses/30/lessons/42885)
        - 선택한 두 수의 합이 X 이하가 되는 쌍을 최대한 많이 만드는 문제
    - [기능개발](https://gaussian37.github.io/interview-ps-p42586/)(https://programmers.co.kr/learn/courses/30/lessons/42586)
        - 남은 작업 시간 및 순차적인 작업 순서를 고려하여 작업의 쌍을 정하는 문제
    - [괄호](https://gaussian37.github.io/interview-ps-9012/)(https://www.acmicpc.net/problem/9012)
        - 괄호가 올바른 짝(여는 괄호와 닫는 괄호 셋)으로 잘 맞추어져 있는지 확인하는 문제 
    - [쇠막대기](https://gaussian37.github.io/interview-ps-p42585/)(https://programmers.co.kr/learn/courses/30/lessons/42585)
        - 스택 자료 구조를 이용하여 쇠막대기가 몇 등분 되는지 확인하는 문제
        - 괄호 구조는 스택을 이용하여 풀면 쉽게 해결할 수 있음
    - [탑](https://gaussian37.github.io/interview-ps-p42588/)(https://programmers.co.kr/learn/courses/30/lessons/42588)
        - 스택 자료구조를 이용하여 효율적으로 문제 해결
    - [주식가격](https://gaussian37.github.io/interview-ps-p42584/)(https://programmers.co.kr/learn/courses/30/lessons/42584)
        - 스택 자료구조를 이용하여 현재 인덱스 값 보다 값이 작아지는(현재 인덱스 보다 오른쪽에 있는 값 중) 최초 시점 구하는 문제
    - [더 맵게](https://gaussian37.github.io/interview-ps-p42626/)(https://programmers.co.kr/learn/courses/30/lessons/42626)
        - 우선순위 큐를 이용하여 최소값을 계속 찾는 문제    
    - [라면공장](https://gaussian37.github.io/interview-ps-p42629/)(https://programmers.co.kr/learn/courses/30/lessons/42629)
        - 자원의 양이 0이하로 떨어지지 않도록 계속 유지 하기 위해 최소한의 횟수로 공급해 주어야 하는 문제
        - 자원을 공급을 할 수 있는 시점과 공급양이 주어질 때 우선순위 큐를 이용하여 최소한의 횟수를 공급하는 기준을 만들 수 있는 아이디어가 필요함
    - [단속카메라](https://gaussian37.github.io/interview-ps-p42884/)(https://programmers.co.kr/learn/courses/30/lessons/42884)
        - 여러 구간이 주어질 때, 구간들의 교집합이 가장 많이 발생하는 구간들을 카운트 하는 문제
    - [에디터](https://gaussian37.github.io/interview-ps-b1406/)(https://programmers.co.kr/learn/courses/30/lessons/b1406)
        - 스택을 이용하여 문자를 추가할 위치를 효율적으로 추적하는 문제. 커서 기준 왼쪽 오른쪽 양측을 관리해야 하므로 stack 2개를 이용하면 쉽게 풀 수 있음
        - 스택을 이용한 이유는 커서 기준 양쪽의 top만 보면 되기 때문입니다.     
            
<br>
    
- `다이나믹 프로그래밍`
    - [땅따먹기](https://gaussian37.github.io/interview-ps-p12913/)(https://programmers.co.kr/learn/courses/30/lessons/12913)
        - 행렬에서 0행부터 끝행까지 지나가면서 점수가 최대가 되도록 만드는 dp문제
    - [등굣길](https://gaussian37.github.io/interview-ps-p12913/)(https://programmers.co.kr/learn/courses/30/lessons/12913)
        - 격자 무늬에서 갈 수 있는 경우의 수를 찾는 문제. 중간에 갈 수 없는 영역이 추가됨
        
<br>

- `완전탐색`
    - [모의고사](https://gaussian37.github.io/interview-ps-p42840/)(https://programmers.co.kr/learn/courses/30/lessons/42840)
        - 문자열 단순 반복 탐색 하면서 일치하는 갯수 카운트 하는 문제
    - [조이스틱](https://gaussian37.github.io/interview-ps-p42860/)(https://programmers.co.kr/learn/courses/30/lessons/42860)
        - 조이스틱으로 원하는 문자열을 만들 수 있는 최소 조이스틱 사용 횟수 
        - 좌/우 움직임을 고려하여 완전탐색을 하는 방법으로 해결
    - [타켓 넘버](https://gaussian37.github.io/interview-ps-p43165/)(https://programmers.co.kr/learn/courses/30/lessons/43165)
        - 깊이우선 탐색으로 완전 탐색하여 가능한 경우의 수 찾는 문제
    - [숫자야구](https://gaussian37.github.io/interview-ps-p42841/)(https://programmers.co.kr/learn/courses/30/lessons/42841)
        - 가능한 모든 숫자의 경우를 기준으로 조건을 모두 만족하는 숫자를 찾는 문제
        - 인풋의 범위를 보았을 때 가능한 모든 숫자의 갯수가 많지 않으므로 숫자 하나 하나를 컴퓨터 연산속도를 이용하여 완전탐색 할 수 있음
    - [소수찾기](https://gaussian37.github.io/interview-ps-p42839/)(https://programmers.co.kr/learn/courses/30/lessons/42839)
        - 가능한 모든 수를 만들어서 소수인지 아닌지 판단하는 문제
        - 가능한 모든 수를 만드는 방법과 소수를 판단하는 방법이 중요함        
        
<br>

- `그래프`
    - [카카오 프렌즈 컬러링북](https://gaussian37.github.io/interview-ps-p1829/)(https://programmers.co.kr/learn/courses/30/lessons/1829)
        - board에서 connected componets의 갯수를 찾는 문제
    - [단어 변환](https://gaussian37.github.io/interview-ps-p43163/)(https://programmers.co.kr/learn/courses/30/lessons/43163)
        - 주어진 단어 리스트를 가지고 인접 리스트를 만든 다음 bfs로 최단 거리를 찾는 문제
        - 한 단어 차이만 나는 경우 단어간 연결되어 있다고 판단할 수 있으므로 인접 리스트를 만드는 작업이 핵심이 되는 문제     
<br>

- `수학`
    - [124 나라의 숫자](https://gaussian37.github.io/interview-ps-p12899/)(https://programmers.co.kr/learn/courses/30/lessons/12899)
        - 진법 변환 관련 문제
        
<br>

- `이분법`
    - [예산](https://gaussian37.github.io/interview-ps-p43237/)(https://programmers.co.kr/learn/courses/30/lessons/43237)
        - 이분법을 이용하여 예산의 상한 가격을 구하는 문제        
    - [입국심사](https://gaussian37.github.io/interview-ps-p43238/)(https://programmers.co.kr/learn/courses/30/lessons/43238)
        - 이분법을 이용하여 모든 사람이 작업을 다 처리할 수 있는 시간의 최소값을 구하는 문제
- `트리`
- `문자열`

<br>

## 면접에는 안나올 것 같은 고급 알고리즘 문제 코드 및 설명

<br>

## 프로그래머스 예제 풀이

- 해시
    - 완주하지 못한 선수
    - 위장
    - 베스트앨범
- 힙    
    - 디스크 컨트롤러
    - 이중우선순위큐
- 완전탐색
    - 소수찾기
    - 숫자야구
    - 카펫
- 깊이/너비 우선 탐색     
    - 단어변환
    - 여행경로
    
