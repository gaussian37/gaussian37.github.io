---
layout: post
title: C++에서 문제 풀 때 좋은 STL
date: 2019-06-02 00:00:00
img: interview/ps/ps.png
categories: [interview-ps] 
tags: [ps, c++] # add tag
---

### `lower_bound`, `upper_bound`

+ lower_bound와 upper_bound를 사용하면 **정렬**된 배열에서 필요한 값을 이분 탐색으로 찾을 수 있습니다.

```
int main() {

	vector<int> v;
	int a[5] = { 1, 2, 2, 2, 3 };
	for (int i = 0; i < 5; i++) {
		v.push_back(a[i]);
	}
	int x = 2;
	int b = (int)(upper_bound(v.begin(), v.end(), x) - lower_bound(v.begin(), v.end(), x));
	int c = lower_bound(v.begin(), v.end(), x) - v.begin();
	int d = v.end() - upper_bound(v.begin(), v.end(), x);

	cout << "2의 갯수 : " << b << '\n';
	cout << "2보다 작은 숫자의 갯수 : " << c << '\n';
	cout << "2보다 큰 숫자의 갯수 : " << d << '\n';
}

```

