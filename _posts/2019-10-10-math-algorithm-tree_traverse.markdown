---
layout: post
title: 트리 순회 (전위 순회, 중위 순회, 후위 순회)
date: 2019-10-11 00:00:00
img: math/algorithm/tree_traverse/0.png
categories: [math-algorithm] 
tags: [algorithm, tree, tree traverse, 트리 순회] # add tag
---

<br>

[알고리즘 관련 글 목차](https://gaussian37.github.io/math-algorithm-table/)

<br>

- 참조 : https://m.blog.naver.com/rlakk11/60159303809

<br>

- 트리 관련 대표적인 문제로 트리 순회가 있습니다. 이번 글에서는 전위 순회, 중위 순회, 후위 순회에 대하여 간단하게 알아보도록 하겠습니다.
- 전/중/후위에 해당하는 순서는 각각 `root`를 먼저/중간/나중에 방문한다는 의미를 가집니다.

<br>
<center><img src="../assets/img/math/algorithm/tree_traverse/0.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 트리 그래프를 전위, 중위, 후위 순서로 탐색해 보겠습니다. 출발은 루트인 0에서 시작하겠습니다.
- `전위 순회` : `root` → left → right 순서로 방문합니다. 
- 위 예제 탐색 순서 : 0 → 1 → 3 → 7 → 8 → 4 → 9 → 10 → 2 → 5 → 11 → 6

<br>

```c
void preorder_traverse(node *t){
	visit(t);
	preorder_traverse(t->left);
	preorder_traverse(t->right);
}
```

<br>

- `중위 순회` : left → `root` → right 순서로 방문합니다.
- 위 예제 탐색 순서 : 7 → 3 → 8 → 1 → 9 → 4 → 10 → 0 → 11 → 5 → 2 → 6

<br>

```c
void inorder_traverse(node *t){
	inorder_traverse(t->left);
	visit(t);
	inorder_traverse(t->right);
}
```

<br>

- `후위 순회` : left → right → `root` 순서로 방문합니다.
- 위 예제 탐색 순서 : 7 → 8 → 3 → 9 → 10 → 4 → 1 → 11 → 5 → 6 → 2 → 0

<br>

```c
void postorder_traverse(node *t){
	postorder_traverse(t->left);
	postorder_traverse(t->right);
	visit(t);
}
```

<br>

[알고리즘 관련 글 목차](https://gaussian37.github.io/math-algorithm-table/)

<br>

