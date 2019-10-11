---
layout: post
title: Trie
date: 2019-10-11 00:00:00
img: math/algorithm/trie/trie.png
categories: [math-algorithm] 
tags: [algorithm, trie] # add tag
---

- 이번 글에서는 문자열 알고리즘의 하나인 `trie`에 대하여 알아보도록 하겠습니다.
- `trie` 관련 알고리즘을 익혔으면 [링크](https://gaussian37.github.io/interview-ps-ps-table/)에서 `trie` 관련 문제를 찾아 한번 풀어보시길 바랍니다.

<br>

## **목차**

<br>

- ### 요약
- ### Trie 알고리즘의 필요성
- ### Trie 알고리즘
- ### C++을 이용한 Trie 알고리즘 구현

<br>

## **요약**

<br>

- 현상 및 문제점:
- 원인: 
- 대책: 
- 방법: 

<br>

## **Trie 알고리즘의 필요성**

<br>

- `Trie` 알고리즘은 문자열 비교를 할 때 효율적으로 사용할 수 있습니다.
- 숫자를 비교할 때에는 `O(1)`의 상수시간 만에 비교할 수 있습니다. 반면 문자열을 비교할 때에는 `O(길이)`의 시간이 걸립니다.
- 그러면 N개의 숫자를 담고 있는 BST(Binary Search Tree)에서는 특정 숫자를 찾는 데 `O(logN)`의 복잡도를 가지지만 문자열 N개를 담고 있는 BST에서는 특정 문자열을 검색하는 데 걸리는 시간은 `O(길이 * logN)`입니다.
- BST를 만드는 시간 복잡도를 살펴보면 숫자의 경우에는 `O(NlogN)`인 반면에 문자열의 경우 `O(길이*NlogN)`이 됩니다. 생각보다 비효율적이게 됩니다.
- 비교를 할 때에는 트리 구조를 만드는 것이 효율적이지만 단순한 트리 구조를 이용하여 문자열을 비교하면 꽤나 비효율적이 됩니다.
- 이런 문제를 개선하기 위하여 등장한 것이 `Trie` 알고리즘입니다.

<br>

## **Trie 알고리즘**

<br>
<center><img src="../assets/img/math/algorithm/trie/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 단어 셋이 {A, to, tea, ted, ten, i, in, inn} 이 있다고 가정하고 Trie 구조의 트리를 만들어 보면 위와 같습니다.
- 이 트리에서 depth는 문자열의 길이를 뜻하게 됩니다. 
- `trie`의 이런 구조로 인하여 또 다른 이름으로 `prefix tree`라고도 하는데 그 이유는 부모 노드가 자식 노드의 prefix가 되기 때문입니다.
- 여기서 발생하는 문제점 한가지는 위 그림의 노드 중에 **te**라는 노드는 실제 단어셋에 존재하지는 않지만 노드로 구성되었습니다.
- 따라서 이 문제를 해결하기 위하여 각 노드에 이 노드가 실제 단어 셋에 있는 단어인지 아닌지를 표시해주는 `flag`가 필요합니다. 이것을 `valid`라고 하고 `valid = true`이면 실제 단어셋에 존재하는 문자열이라고 표현해줍니다. 

<br>

## **C++을 이용한 Trie 알고리즘 구현**

<br>