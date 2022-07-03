---
layout: post
title: PCD와 Depth Map의 변환 관계 정리
date: 2022-06-18 00:00:00
img: vision/depth/pcd_depthmap/0.png
categories: [vision-depth] 
tags: [vision, depth, point cloud, depth map] # add tag
---

<br>

- 이번 글에서는 `Point Cloud`와 `Depth Map` 사이의 변환 방법에 대하여 알아보도록 하겠습니다.
- 먼저 라이다를 통해 취득한 `PCD(Point Cloud Data)`를 이미지에 Projection 하여 `Depth Map`을 만드는 방법에 대하여 다루어보고 `Depth Map`이 있을 때, 이 값을 `PCD` 형태로 나타내는 방법에 대하여 다루어 보도록 하겠습니다.

<br>

## **목차**

<br>

- ### PCD to Depth Map 원리
- ### PCD to Depth Map 실습
- ### Depth Map to PCD 원리
- ### Depth Map to PCD 실습

<br>

- 