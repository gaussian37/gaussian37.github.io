---
layout: post
title: Depth Estimation의 Metric
date: 2021-03-01 00:00:00
img: vision/depth/metrics/0.png
categories: [vision-depth] 
tags: [depth estimation, metrics, rel, rmse] # add tag
---

<br>

[Depth Estimation 관련 글 목차](https://gaussian37.github.io/vision-depth-table/)

<br>

- 참조 : http://ylatif.github.io/papers/IROS2016_ccadena.pdf

<br>

- 이번 글에서는 Depth Estimation 논문에서 성능 지표로 사용하는 metric에 대하여 간략하게 알아보도록 하겠습니다.
- 주로 사용하는 Metric은 `Absolute Relative Error`, `Linear Root Mean Square Error`, `log scale invariant RMSE`, `Accuracy under a threshold`



<br>

```python
def compute_depth_errors(gt, pred):   

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    delta = torch.max((gt / pred), (pred / gt))
    a1 = (delta < 1.25     ).float().mean()
    a2 = (delta < 1.25 ** 2).float().mean()
    a3 = (delta < 1.25 ** 3).float().mean()

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
```




<br>

[Depth Estimation 관련 글 목차](https://gaussian37.github.io/vision-depth-table/)

<br>