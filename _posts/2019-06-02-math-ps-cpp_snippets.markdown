---
layout: post
title: C++에서 문제 풀 때 좋은 코드 snippets
date: 2019-06-02 00:00:00
img: math/ps/ps.png
categories: [math-ps] 
tags: [ps, c++] # add tag
---

<br>

## **목차**

<br>

- ### [문자열 입출력](#문자열-입출력-1)
- ### [문자열에서 숫자 찾는 tokenizer](#문자열에서-숫자-찾는-tokenizer-1)
- ### [문자열 숫자와 정수,실수형 숫자 간의 변환](#문자열-숫자와-정수실수형-숫자-간의-변환-1)
- ### [lower_bound, upper_bound](#lower_bound-upper_bound-1)
- ### [unique와 erase 사용하기](#unique와-erase-사용하기-1)

<br>

## **문자열 입출력**

<br>

- C++을 사용하더라도 `printf`, `scanf`를 여전히 사용할 수 있고 입출력 속도 또한 cin, cout보다 빠르므로 printf, scanf를 사용하는 것을 추천드립니다.
- 다만, C++의 cin, cout을 scanf, print와 같이 빠르게 사용하고 싶다면 다음 코드를 입력해 줍니다.

<br>

```cpp
ios_base::sync_with_stdio(false);
cin.tie(NULL);
cout.tie(NULL);
```

<br>

- 대신에 `cin`, `cout`을 쓰면서 절대로 `scanf`와 `printf`를 같이 써서는 안됩니다. (동기화가 끊겨서 값이 뒤죽박죽 됩니다.)
    
<br>

- 입력을 EOF까지 입력을 받는 경우 다음과 같이 받으면 됩니다.
    - C 스타일 : `while(scanf("%d %d", &a, &b) == 2)`
    - C++ 스타일 : `while(cin>>a>>b))`

<br>

- **new line을 받을 때 까지 한 줄을 입력 받고 싶으면** 다음과 같이 입력 받습니다.
- `C 스타일` : `scanf("%[^\n]\n", s);`
	- scanf 안의 `%[^\n]`의 뜻을 살펴보면 **^**뒤의 문자만 빼고 입력을 받겠다는 뜻입니다. 
	- 즉, 개행 문자는 빼고 문자를 받는 다는 뜻이고 대괄호 밖의 마지막 문자 개행문자의 뜻은 입력 받는 마지막의 문자는 개행 문자여야 한다는 뜻입니다.
	- 따라서 입력 받는 문자열에서 마지막 문자는 개행문자이어야 하고 그 이전에 받는 개행 문자는 무시한다는 뜻입니다.
- `C++ 스타일` : `getline(cin, s);`

<br>

- `scanf`에서 `%`와 데이터 타입 `d`, `s` 등의 사이에 숫자를 입력하면 그 숫자에 해당하는 길이 만큼 입력을 받게됩니다.

<br>

```cpp
int x;
scanf("%1d", &x);

char s[100];
scanf("%10s",s);
```

<br>

- `%d` 사이에 숫자를 넣으면, 그 길이 만큼 입력을 받게 됩니다. 예를 들어 `%1d`를 사용하고 12345를 입력 받으면 1, 2, 3, 4, 5를 따로 따로 입력받을 수 있습니다.
- `%s`의 경우도 갯수를 지정해서 입력받을 수 있습니다. 만약 입력받을 수 있는 것의 갯수가 지정한 갯수보다 적으면 그만큼만 입력을 받게됩니다.

<br>

## **문자열에서 숫자 찾는 tokenizer**

<br>

- `tokenize`는 문자열 숫자를 받아서 del 기준으로 token 하여 문자열 벡터로 반환합니다.
- `tokToNum`은 문자열 숫자를 받아서 del 기준으로 token 하여 숫자 벡터로 반환합니다.

<br>

```cpp
vector<string> tokenize(string s, string del = " ") {
	vector<string> ret;
	for (int i = 0, j; i < s.size(); i = j + 1) {
		if ((j = s.find_first_of(del, i)) == -1) j = s.size();
		if (j - i > 0) ret.push_back(s.substr(i, j - i));
	}
	return ret;
}

vector<int> tokToNum(string s, string del = " ") {
	vector<int>ret;
	vector<string> vs = tokenize(s, del);
	for (int i = 0; i < vs.size(); ++i) ret.push_back(stoi(vs[i]));
	return ret;
}

```

<br>

## **문자열 숫자와 정수,실수형 숫자 간의 변환**

<br>

- from `문자열` to `정수`
    - stoi : string → int
    - stol: string → long
    - stoll : string → long long
    - stof : string → float
    - stod : string → double
    - stold : string → long double
    - stoul : string → unsigned long
    - stoull : string → unsigend long long
- from `정수, 실수` to `문자열`
    - to_string() 함수 사용

<br>

## **lower_bound, upper_bound**

<br>

- lower_bound와 upper_bound를 사용하면 **정렬**된 배열에서 필요한 값을 이분 탐색으로 찾을 수 있습니다.

<br>

```
int main() {

	vector<int> v;
	int a[5] = { 1, 2, 2, 2, 3 };
	for (int i = 0; i < 5; i++) {
		v.push_back(a[i]);
	}
	int x = 2;
	int b = upper_bound(v.begin(), v.end(), x) - lower_bound(v.begin(), v.end(), x);
	int c = lower_bound(v.begin(), v.end(), x) - v.begin();
	int d = v.end() - upper_bound(v.begin(), v.end(), x);

	cout << "2의 갯수 : " << b << '\n';
	cout << "2보다 작은 숫자의 갯수 : " << c << '\n';
	cout << "2보다 큰 숫자의 갯수 : " << d << '\n';
}

```

<br>

## **unique와 erase 사용하기**

- unique를 사용하면 중복된 데이터들을 전부 뒤쪽으로 옮깁니다.
    - 이 때 unique에서 반환하는 값은 iterator로 중복되는 값의 시작 주소를 나타냅니다.
    - `unique(v.begin(), v.end())`
- erase를 사용하면 원소를 삭제할 수 있습니다.
    - `v.erase(v.begin(), v.end())`
- 따라서 `v.erase(unique(v.begin(), v.end()), v.end())`를 사용하면 
    - 먼저 **unique**를 이용하여 유니크한 값만 앞쪽으로 모으로 중복된 값을 뒤쪽으로 모은 다음에 중복값의 시작 위치 iterator를 반환합니다.
    - 중복된 값의 시작 위치 ~ 끝까지를 erase를 통하여 삭제합니다.
     
<br>
    
