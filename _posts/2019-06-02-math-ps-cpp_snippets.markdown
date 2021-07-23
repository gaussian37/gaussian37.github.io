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

- ### lower_bound, upper_bound
- ### unique와 erase 사용하기
- ### 문자열에서 숫자 찾는 tokenizer
- ### 문자열 숫자와 정수,실수형 숫자 간의 변환

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