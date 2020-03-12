---
layout: post
title: latency와 throughput 이란
date: 2019-10-27 00:00:00
img: c/linux/os/os.jpg
categories: [c-linux] 
tags: [os, operating system] # add tag
---

<br>

- latency (**시간** 단위) : 어떤 작업을 처리하는 데 걸리는 시간
- throughput (**일** 단위) : 초당 처리하는 작업의 갯수

<br>

- 기본적으로 latency와 throughput은 나타내는 단위가 다릅니다. latency는 작업이 걸리는 시간을 의미하는 데 반해 throughput은 초당 처리하는 작업의 수를 나타내기 때문입니다.
- 따라서 latency는 작을수록 좋은것이고 throughput은 클 수록 좋은 것입니다.
- 처리하는 프로세서의 자원은 한정되어 있기 때문에 latency 낮추려면 즉, 어떤 작업을 최대한 빨리 끝내려고 한다면 자원을 그 작업에 최대한 많이 할당해야 합니다. 그러면 latency를 낮출 수 있습니다.
- 반면 어떤 작업에 자원이 몰리게 되면 한번에 할 수 있는 작업의 갯수는 줄어들게 됩니다. 이는 즉 throughput에 영향을 주게 됩니다.
- 따라서 latency와 throughput은 서로 영향을 주게 됩니다.

<br>

<br>
<center><img src="../assets/img/c/linux/latency_and_throughput/0.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 예를 들어 SCV 10개가 있는데, 해야할 작업이 ① 미네랄 1,000모으기와 ② 배럭 5개 짓기라고 가정해 보겠습니다.
- 각 작업의 latency를 낮추려면 미네랄 1,000모으기는 SCV 10개를 모두 동원하는 것이 좋습니다. 물론 배럭 5개 짓기도 5개 SCV를 이용하여 동시에 짓는 것이 latency를 줄일 수 있습니다.
- throughput의 관점에서 보면 어떨까요? latency를 줄이기 위해서 한 작업에 SCV를 다 배정하는 것이 효율적일까요? 그것은 어떤 지점이 최적인 지 확인을 해봐야 합니다. throughput의 관점에서는 동시에 많은 일을 하는 것이 좋습니다. (물론 경우에 따라서는 한 작업, 한 작업씩 차례대로 끝내는게 나을 수도 있겠지만요..)
- 여기서 중요한 것은 **한정된 자원으로 인하여** lantency를 줄이는 것과 throughput을 최대로 한다는 것 이것은 서로 영향을 준다는 것입니다.
- 관점을 바꿔서 보면 `작업`의 입장에서는 latency가 낮아지길 원합니다. 빨리 처리되기를 바라는 것입니다. 반면 `시스템`의 입장에서는 throughput이 커지길 원합니다. 즉, 전체적으로 최적의 상태로 작업이 처리되길 원하는 것입니다.

<br>

- `latency`와 `throughput`의 가장 효율적인 최적점을 찾는 것은 네트워크 프로그래밍이나 운영체제 문제에서 아주 중요한 문제입니다.
- 따라서 이 두 지표를 잘 관리하는 것이 효율적인 프로그래밍을 하는 것의 기준이 될 수 있으니 참조 하시기 바랍니다.
