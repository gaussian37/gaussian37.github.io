---
layout: post
title: 자율주행을 위한 라이다(Lidar) 센서와 라이다 포인트 클라우드 처리 방법
date: 2021-01-10 00:00:00
img: autodrive/lidar/intro/0.png
categories: [autodrive-lidar] 
tags: [autonomous drive, 자율 주행, 라이다, lidar, open3d, RANSAC, DBSCAN, KDTree] # add tag
---

<br>

- 참조 : REC.ON : Autonomous Vehicle (FastCampus)
- 참조 : http://www.kibme.org/resources/journal/20210513134622527.pdf

<br>

- 이번 글에서는 라이다 관련된 내용을 종합적으로 정리하기 위한 글입니다. 다소 중복된 내용이 여러번 언급될 수 있으니 참조하여 글을 읽어주시면 됩니다.

<br>

## **목차**

<br>

- ## **라이다 포인트 클라우드 처리 방법**
- ### [라이다와 포인트 클라우드](#라이다와-포인트-클라우드-1)
- ### [Processing : open3d를 이용한 포인트 클라우드 처리](#open3d를-이용한-포인트-클라우드-처리)
- ### [Voxel Grid Downsampling](#voxel-grid-downsampling-1)
- ### [RANSAC을 이용한 도로와 객체 구분](#)
- ### [DBSCAN을 이용한 포인트 클라우드 클러스터링](#)
- ### [HDBSCAN을 이용한 포인트 클라우드 클러스터링](#)
- ### [PCA를 이용한 객체 Boundgin Box](#)

<br>

- ## **자율주행에서의 라이다 동향**
- ### [자율주행 요소 기술](#자율주행-요소-기술-1)
- ### [라이다 센서](#라이다-센서-1)
- ### [라이다의 활용 : 위치 추정, 객체 인지](#)

<br>

- 먼저 이번 글의 첫번째 목적은 아래와 같은 5가지의 프로세스를 하나씩 접근해 보는 것입니다.

<br>
<center><img src="../assets/img/autodrive/lidar/intro/14.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- ① `Sense` : 라이다 센서는 빛 (레이저)를 송신하고 주변 물체로 부터 반사되어 다시 수신하여 물체의 위치를 확인합니다. 하드웨어적인 상세 내용은 글 후반부의 `자율주행에서의 라이다 동향`에서 부터 살펴보시길 바랍니다.
- ② `Generate a File` : 라이다 센서를 통하여 인식한 결과를 통해 raw data를 생성합니다. 공개된 라이다 데이터셋은 보통 가공이 되어 있는 데이터인 반면에 직접 라이다를 통해 얻은 데이터는 노이즈도 섞여있고 포인트가 필요 이상으로 많이 있기도 합니다. 
- ③ `Process the File` : 라이다 데이터를 읽으면 포인트 클라우드 형태로 되어있습니다. 3D에서 포인트 클라우드 형태로 데이터를 읽기 위해서 `open3D` 라는 패키지를 사용할 예정입니다.
- ④ `Process Point Cloud` : 라이다 데이터에서 의미 있는 정보를 찾기 위하여 포인트를 샘플링 하기도하고 추가적인 알고리즘을 이용하여 포인트 들을 클러스터링을 하기도 합니다. 이번 글에서는 bounding box를 적용하여 물체를 찾는 간단한 방법을 다루어 볼 예정입니다.
- ⑤ `Visualize the results` : 포인트 클라우드 처리가 끝난 결과를 시각화해서 보는 단계를 의미합니다. 위 그림과 같이 포인트 클라우드 상에서 bounding box가 잘 생성되었는 지 살펴보도록 하겠습니다.

## **라이다와 포인트 클라우드**

<br>

<br>
<center><img src="../assets/img/autodrive/lidar/intro/15.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 포인트 클라우드는 대표적으로 위 그림과 같이 6개의 포맷으로 저장될 수 있습니다.
- 공통적으로 가지고 있는 정보는 3차원 공간 상에서 `XYZ 좌표` 정보이며 추가적으로 `intensity` 정보를 많이 사용합니다. 
- 라이다로 취득한 3차원 포인트 클라우드에서는 가까운 물체가 멀리 있는 물체에 비해 조밀하게 샘플링 되는 특징이 있습니다. 또한 멀리 있는 물체일수록 반사되어 돌아오는 신호의 강도가 약해지고 큰 노이즈를 포함하기 쉽습니다. 이러한 성질을 이용하여 라이다를 이용하여 얻은 포인트 클라우드에는 3차원 좌표와 함께 **반사되어 돌아온 신호의 강도**를 나타내는 반사도의 세기 정보가 포함되는데 이 정보를 `intensity` 라고 합니다.

<br>
<center><img src="../assets/img/autodrive/lidar/intro/16.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림의 왼쪽에서는 포인트 클라우드의 색이 녹색과 파란색으로 비교적 반사도 세기가 낮고 자동차 번호판 부분은 붉은색으로 비교적 높은 세기를 가집니다. 반사도 세기에 영향을 미치는 요인들은 여러가지가 있는데 레이저 펄스가 통과하는 매질이나 물체의 표면과 펄스가 만나는 각도, 대상 물체 표면의 반사율에 따라서 돌아오는 신호의 세기가 결정됩니다.
- 앞의 비행기 포인트 클라우드에서도 이와 같은 intensity가 반영되어 색으로 표현되어 있음을 알 수 있습니다.

<br>
<center><img src="../assets/img/autodrive/lidar/intro/17.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 포인트 클라우드를 저장할 때, 대표적으로 ASCII 형식과 Binary 형식을 사용합니다. 잘 알려진 바와 같이 ASCII는 바로 텍스트에서 읽을 수 있지만 용량이 굉장히 커진다는 단점이 있습니다.
- 반면 Binary 파일은 텍스트 형태로 읽을 수 없지만 좀 더 compact 하여 용량이 작고 더 많은 정보를 가질 수 있고 읽을 때 더 빠르게 읽을 수 있습니다. 따라서 Binary 형태로 저장하고 읽어서 쓰는 것이 일반적입니다. 앞으로 Binary 파일은 bin 파일로 명칭하겠습니다.

<br>
<center><img src="../assets/img/autodrive/lidar/intro/18.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- ① `Processing` : bin 파일을 읽으면 1번과 같은 형태의 많은 포인트 클라우드가 생성됩니다. 
- ② `Downsampling` : 따라서 2번과 같이 Downsamping을 통하여 중복된 포인트 클라우드 수를 줄여줍니다.
- ③ `Segmentation` : Downsampling된 포인트 들을 의미 단위로 나누는 작업을 합니다. 예를 들어 어떤 포인트는 바닥이고 어떤 포인트는 벽인지 등을 구분합니다.
- ④ `Clustering` : Segmentation 작업을 통하여 클래스가 분류되면 거리를 기반으로 점들을 클러스터링 하여 필요한 객체 단위로 묶습니다.
- ⑤ `Bounding Box` : Segmentation과 Clustering 과정을 통해 점들이 클래스와 거리 별로 구분이 되게 되면 Bounding Box를 사용하여 객체의 위치를 표시할 수 있습니다.
- 이와 같은 ① ~ ⑤ 가 흔히 사용하는 `Obstacle Detection Process` 방법입니다. **이번 글에서도 이와 같은 방식을 이용하여 어떻게 라이다 포인트 클라우드를 처리하는 지 살펴보도록 하겠습니다.**

<br>

- 라이다 데이터를 다루기 위해서는 라이다를 다루기 위한 포인트 클라우드 라이브러리와  데이터셋이 필요합니다. 아래 링크를 통해 현재 많이 사용하는 라이다 포인트 클라우드 처리를 위한 라이브로리와 데이터셋을 확인할 수 있습니다.
- `Point Cloud Libraries` : https://github.com/openMVG/awesome_3DReconstruction_list#mesh-storage-processing
- `LiDAR Datasets` : https://boschresearch.github.io/multimodalperception/dataset.html

<br>

## **Processing : open3d를 이용한 포인트 클라우드 처리**

<br>

- 포인트 클라우드를 처리하기 위하여 3D 포인트 클라우드를 시각화해서 보기 위해 `python`에서 `open3d` 라이브러리를 사용해보도록 하겠습니다. 추가적으로 numpy, matplotlib, pandas 또한 설치하여 데이터를 다루는 데 활용하고 `PPTK` 패키지를 설치하여 더 개선된 시각화를 사용해 보도록 하겠습니다.

<br>

- `pip install open3d`
- `pip install numpy`
- `pip install matplotlib`
- `pip install pandas`
- `pip install pptk`

<br>

- `open3d`의 documentation은 아래 링크에서 확인할 수 있습니다.
    - 링크 : http://www.open3d.org/docs/release/
- 기본적인 `open3d` 사용법에 대한 튜토리얼은 다음 링크에서 확인할 수 있습니다.
    - 링크 : http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html

<br>
<center><img src="../assets/img/autodrive/lidar/intro/19.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림과 같이 `open3d`나 `pptk` 모두 라이다 포인트 클라우드를 읽는 데 문제는 없습니다. 다만 보시는 것과 같이 `pptk`가 좀 더 사용자 편하게 보여주는 점 참조하시면 도움이 됩니다.

<br>

- 이번 글에서 사용할 데이터는 `KITTI`에서 제공하는 `data_object_velodyne` 데이터로 약 29 GB 용량을 가지는 데이터 입니다.
- [KITTI 사이트](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)에 접속하시고 **Download Velodyne point clouds, if you want to use laser information (29 GB)** 항목을 클릭하셔서 다운받으시면 됩니다.
- 위 데이터는 `bin` 형식의 파일이고 아래에 설명할 `convert` 파일을 통하여 `bin → PCD`로 포맷 변환이 가능합니다. `bin`은 효율적으로 데이터를 저장하고 읽기 위해 이와 같이 저장하고 실제 사용할 때에는 `PCD` 라는 확장자의 파일을 사용하는 것이 일반적입니다.

<br>

```python
import argparse
import open3d as o3d
import numpy as np
import struct
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', required = True)
    parser.add_argument('--dest_path', required = True)
    parser.add_argument('--size_float', default = 4, type=int)
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    
    args = get_args()  
        
    if os.path.exists(args.dest_path) == False:
        os.makedirs(args.dest_path)
    
    files_to_open = os.listdir(args.src_path)
    for file_to_open in files_to_open:
        
        print("write : ", file_to_open)
        # list_pcd = []
        # with open (args.src_path + os.sep + file_to_open, "rb") as f:
        #     byte = f.read(args.size_float * 4)
        #     while byte:
        #         x,y,z,intensity = struct.unpack("ffff", byte)
        #         list_pcd.append([x, y, z])
        #         byte = f.read(args.size_float*4)
        # np_pcd = np.asarray(list_pcd)
        np_pcd = np.fromfile(args.src_path + os.sep + file_to_open, dtype=np.float32).reshape((-1, 4))
        np_pcd = np_pcd[:, :3]
        pcd = o3d.geometry.PointCloud()
        v3d = o3d.utility.Vector3dVector
        pcd.points = v3d(np_pcd)

        file_to_save = args.dest_path + os.sep + os.path.splitext(file_to_open)[0] + ".pcd"
        
        o3d.io.write_point_cloud(file_to_save, pcd)
```

<br>

- 위 코드를 이용하면 폴더 내의 모든 `bin` 파일을 `pcd` 파일로 변경합니다. 파이썬 코드 사용은 다음과 같습니다. (위 코드를 `bin_to_pcd.py` 로 저장했다고 가정합니다.)

<br>

```python
python bin_to_pcd.py --src_path=/home/desktop/data_object_velodyne/training/velodyne --dest_path=/home/desktop/data_object_velodyne/training/velodyne_pcd
```

<br>

- 저장된 `pcd` 파일은 다음 코드를 통해 읽은 후 `open3d`나 `pptk`로 시각화 할 수 있습니다.

<br>

```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("000000.pcd")

# visualization with open3d
open3d.visualization.draw_geometries([pcd])

# visualization with pptk
v = pptk.viewer(pcd.points)
```

<br>

- 파이썬에서 `open3d` 의 `open3d.cpu.pybind.geometry.PointCloud` 객체 형태로 파일을 읽게 된다면 이 값은 numpy로 변경할 수 있습니다. numpy로 변경하게 되면 파이썬에서 다양한 목적으로 쉽게 사용할 수 있습니다.
- 다음 코드는 읽은 `pcd` 파일을 numpy로 변경하거나 임의의 numpy 파일을 다시 point cloud 객체로 변환하는 방법입니다.

<br>

```python
import open3d as o3d

# point cloud → numpy
pcd = o3d.io.read_point_cloud("000000.pcd")
pcd_np = np.asarray(pcd.points)
print(pcd_np)
# array([[896.994  ,  48.7601 ,  82.2656 ],
#        [906.593  ,  48.7601 ,  80.7452 ],
#        [907.539  ,  55.4902 ,  83.6581 ],
#        ...,
#        [806.665  , 627.363  ,   5.11482],
#        [806.665  , 654.432  ,   7.51998],
#        [806.665  , 681.537  ,   9.48744]])

# numpy → point cloud
A = np.random.random((1000, 3)) * 1000
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(A)
# open3d.visualization.draw_geometries([pcd])
v = pptk.viewer(pcd.points)
# v = pptk.viewer(np.asarray(pcd.points))
```

<br>
<center><img src="../assets/img/autodrive/lidar/intro/20.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 코드와 같이 pcd → numpy 형태로 변형할 수 있고 반대로 임의의 numpy 값을 pcd로 변형하여 시각화 할 수 있습니다. 단, 이 때, pcd로 변형될 수 있도록 좌표값에 대한 정보를 3열을 이용하여 나타내어야 합니다.
- `pptk.viewer`에 numpy 타입의 데이터를 직접 입력해 주는 방식도 사용 가능합니다.

<br>

## **Voxel Grid Downsampling**

<br>

- 앞에서 살펴본 포인트 클라우드의 수가 중복된 위치에 필요 이상으로 촘촘히 있는 것을 확인할 수 있습니다.
- 흔히 센서를 통해 얻은 데이터는 원본 데이터를 사용하기보다 원본 데이터를 일정한 간격으로 줄이는 downsampling 방법을 통해 처리를 해서 사용합니다.
- 이와 같이 사용하는 이유는 실제 포인트 클라우드를 처리할 때, 효율적으로 처리하기 위함입니다. 원본 데이터를 그대로 사용할 경우 정보는 많으나 데이터를 처리하는 데 시간이 너무 많이 필요하기 때문입니다.
- 이미지에서도 이와 비슷하게 이미지를 downsampling 하여 실제 알고리즘을 처리할 때 필요한 시간을 줄이기도 하는데, 이와 같은 목적입니다.

<br>

- 라이다 포인트 클라우드에서 downsampling 하는 흔한 방법은 `Voxel Grid` 기반의 downsampling 입니다.
- `Voxel Grid`는 3D 상의 cube 이며 `Voxel Grid Downsampling`은 이 Voxel Grid 내부에 점 1개만 유지하고 나머지는 필터해서 제거하는 방식입니다. 즉, Voxel Grid 내부에 점이 여러개 있다고 하더라도 1개만 남고 나머지는 없어지므로 유사한 위치에 과도하게 많은 정보를 제거한다는 뜻이 됩니다.

<br>
<center><img src="../assets/img/autodrive/lidar/intro/22.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림과 같이 1개의 Voxel Grid에서는 1개의 점만 남기고 나머지는 모두 지워버려 downsampling을 합니다. 이 경우 파라미터는 `Voxel Grid`의 크기가 됩니다. 앞으로 다룰 `voxel_size`의 단위는 `meter`입니다. 따라서 0.1m, 0.2m 등과 같은 길이가 많이 사용됩니다.
- `open3d`의 documentation에 따르면 한 개의 `Voxel Grid`에 포함된 포인트 클라우드들을 대상으로 평균을 내어 평균 위치의 포인트를 한 개 생성하고 기존의 포인트 들은 제거하는 방식을 통하여 downsampling 하는 것으로 알려져 있습니다.

<br>
<center><img src="../assets/img/autodrive/lidar/intro/23.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 위 그림은 왼쪽부터 downsampling 하기 이전, 0.2의 voxel size로 downsampling을 한것 그리고 1의 voxel size로 downsampling을 한 것을 차례대로 시각화 한 것입니다.
- voxel size가 커질수록 3D 공간 상에서 필터링 되는 점이 많아지기 때문에 좀 더 sparse 해지게 됩니다.
- 아래 코드의 `voxel_size` 값을 통해 크기를 조절할 수 있습니다.

<br>

```python
print(f"Points before downsampling: {len(pcd.points)} ")
# Points before downsampling: 115384 
pcd = pcd.voxel_down_sample(voxel_size=0.2)
print(f"Points after downsampling: {len(pcd.points)}")
# Points after downsampling: 22625
v = pptk.viewer(pcd.points)
```

<br>

- 추가적인 downsampling 방법으로는 `ROI (Region Of Interest)` 영역을 기반으로 downsampling 하는 방법이 있습니다. 이 방법은 내가 관심있는 영역이 10m 라면 10m 이외의 모든 점은 제거하는 방법입니다. 간단하면서도 내가 필요로 하는 포인트만 사용한다는 점에서 굉장히 효율적인 방법입니다.

<br>

## **Segmentation with RANSAC**

<br>

<br>


<br>

## **자율주행에서의 라이다 동향**

<br>

## **자율주행 요소 기술**

<br>

- 최근들어 `라이다`를 사용해야 하는 지에 대한 유무가 논쟁이 되고 있고 테슬라는 라이다는 물론 레이더도 사용하지 않으려고 하고 있습니다. 전 세계적으로 이런 논쟁이 있지만 `라이다` 센서 자체의 특성이 있기 때문에 이 글에서 한번 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/autodrive/lidar/intro/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 표는 이제 많은 사람들이 알고 계시는 자율주행의 6단계를 나타냅니다. 이번 글에서 다룰 `라이다`는 레벨 2단계 이상에서의 자율주행을 구현하기 위하여 사용되고 있습니다. 일반적으로 라이다를 사용하여 자율주행을 구현할 때, 현재 수준으로는 레벨 3의 양산차 판매 또는 레벨4의 로보택시 서비스를 판매하는 것을 목표로 하고 있습니다.

<br>

- 자율주행을 구현하기 위하여 센서를 통한 `감지`, `주변 객체 인지`, `위치 추정`, `경로 계획 및 제어` 등이 필요합니다.
- 이 기능들을 구현하기 위하여 가장 기본이 되는 센서를 통한 `감지`가 매우 중요하기 떄문에 다양한 센서들이 사용됩니다. 

<br>
<center><img src="../assets/img/autodrive/lidar/intro/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림과 같이 자율주행을 구현하기 위하여 다양한 센서를 사용하고 있습니다. 센서들마다 특성이 있기 때문에 서로 다른 센서가 보완해 주고 있습니다. (현 시점으로 테슬라는 카메라, 초음파 센서만 사용하고 있긴 하지만...)
- 대표적으로 `GPS`, `IMU`, `카메라`, `레이더` 등이 있고 자율주행에서 사용하는 `HD Map` 그리고 이 글에서 다룰 `라이다`가 있습니다. 먼저 `라이다`를 써야하는 이유를 살펴보기 전에 다른 센서의 특성을 간략하게 살펴보고 `라이다`가 가지는 장점을 다루어 보겠습니다.

<br>
<center><img src="../assets/img/autodrive/lidar/intro/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 먼저 `GPS (Global Positioning System)` 센서는 차량의 위치를 추정하는 데 사용되는 센서입니다. 위성으로 부터 신호를 받아서 차량의 절대적인 좌표를 얻게 되는데 문제는 현재 제공되는 GPS의 오차 수준이 미터 단위 정도의 정확도 밖에 제공하고 있지 못하기 때문에 자율 주행에 필요한 수십 cm 단위의 정확도를 달성하기 위해서는 주변에 있는 통신 인프라와 함께 통신하는 DGPS나 GPS-RTK와 같은 기술들이 필요합니다.
- 또한 `GPS`는 고층 빌딩이 둘러 쌓여있거나 지하 터널에서는 순간적으로 신호를 받지 못하는 경우가 발생할 수 있습니다. 모든 판단을 센서를 통하여 하는 자율 주행 시스템의 경우 순간적인 신호 끊김도 위험하기 때문에 GPS 만으로는 위치 추정이 어렵습니다.

<br>
<center><img src="../assets/img/autodrive/lidar/intro/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 그래서 함꼐 사용되는 센서가 `IMU (Inertial Measurement Unit)` 입니다. `IMU`는 순간순간 자동차의 속도, 가속도 및 이동 방향을 **관성 정보**를 이용하여 측정하는 센서입니다. 따라서 순간적인 차량의 이동 정보를 `GPS`와 함께 이용하여 차량의 위치를 추적하는데 사용됩니다.
- `IMU` 센서는 순간 순간 위치를 파악할 수 있으나 매 순간 위치에 센서의 오차가 있을 수 있고 이 오차가 계속 누적이 되면 차이가 발생할 수 있으므로 최근에는 카메라, 레이더, 라이다 등이 같이 사용되게 됩니다.

<br>
<center><img src="../assets/img/autodrive/lidar/intro/5.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- `HD (High-definition) map`이란 단순히 내비게이션에 사용되는 지도 정보 뿐 아니라 자율 주행 차량이 이동하는 도로 환경 정보 (차선, 가드레일 등)를 알수 있으므로 위치를 판단할 때 사용할 수 있습니다.
- 하지만 `HD map`을 구축하는 데 굉장히 많은 비용이 들고 한번 구축했다고 하더라도 지속적인 업데이트가 필요하기 때문에 모든 지역의 HD map을 구축하는 데 한계가 있습니다.

<br>
<center><img src="../assets/img/autodrive/lidar/intro/6.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 자율 주행 자동차 및 ADAS에서 가장 많이 사용되는 `카메라` 같은 경우 풍부한 텍스쳐 정보를 담고 있는 영상 정보를 얻을 수 있습니다.
- 카메라 센서의 한계점으로 지적되는 **주변 환경에 취약하다는 단점**이 있어서 최근에는 레이더나 라이다와 함께 사용되고 있습니다.
- 최근에 딥러닝 기술의 발달로 더욱 높은 정확도로 자율 주행에 필요한 정보를 카메라를 통해 얻고 있는 추세입니다.

<br>
<center><img src="../assets/img/autodrive/lidar/intro/7.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 카메라와 달리 `레이더, RADAR(Radio Detection and Ranging)`는 `전자파`를 이용하여 거리나 속도 정보를 자율 주행 자동차에 제공합니다. 레이더 같은 경우 원거리 물체를 측정하는 데 용이하고 특히 눈, 비, 조명과 같은 주변 환경에 굉장히 강건하여 악조건 환경에 자율 주행 시스템의 판단을 돕는데 사용되고 있습니다.
- 하지만 레이더의 경우 아직 까지 해상도가 낮기 때문에 레이더 센서의 값만을 가지고 감지된 물체가 차량인지 사람인 지 판단하는 데에는 한계가 있습니다. 현재 수준으로는 물체의 구분보다는 물체의 유무 판단에 적합합니다.

<br>
<center><img src="../assets/img/autodrive/lidar/intro/9.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림은 레이더를 통하여 도로 위의 물체를 파악하여 위치를 표시한 예시입니다.
- 지금 까지 라이다 센서를 제외한 자율 주행에 많이 사용되는 센서들의 특성에 대하여 간략하게 알아보았습니다.

<br>

## **라이다 센서**

<br>

- 그러면 이 글의 주제인 라이다에 대하여 알아보도록 하겠습니다.

<br>
<center><img src="../assets/img/autodrive/lidar/intro/8.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- `라이다, LIDAR(Light Detection and Rangin)`는 빛 즉 `레이저`를 이용하여 고해상도의 3차원 정보를 제공하는 센서입니다. 

<br>
<center><img src="../assets/img/autodrive/lidar/intro/12.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림과 같이 라이다는 레이저를 송출하고 레이저가 물체에 맞고 반사되어 돌아오는 시간을 계산하여 빛의 이동 시간을 계산하고 이를 통해 주변 환경에 대한 3차원 형상을 측정합니다.

<br>
<center><img src="../assets/img/autodrive/lidar/intro/13.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 좀 더 상세하게 알아보면 라이다의 프로세는 위 그림과 같습니다.
- ① `Emitter` : 레이저를 송출하는 역할을 합니다.
- ② `Scanning System` : 송출한 레이저를 주변 환경에 맞게 조사하는 역할을 합니다.
- ③ `Receiver` : 반사되어 들어오는 빛을 다시 측정하는 역할을 합니다.
- ④ `Signal Processing` : `Emitter` ~ `Receiver` 까지 걸린 시간을 이용하여 각 포인트 마다의 거리를 계산하는 역할을 합니다.
- ⑤ `Software` : `Signal Processing`을 통해 얻은 각 Point 정보를 이용하여 주변 물체에 대한 측정 결과를 제공합니다.

<br>

- 라이다의 특성 상 카메라에 비해서 눈, 비, 안개, 조명과 같은 주변 환경 등에 강건하기도 하고 레이더에 비해서 고해상도의 3차원 형상 정보를 제공하고 있어서 레이더, 카메라와 함꼐 주변 환경을 감지할 수 있습니다.

<br>
<center><img src="../assets/img/autodrive/lidar/intro/10.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 라이다를 이용하면 위 그림과 같이 주변 환경의 3D 정보를 수많은 점들의 뭉치인 point cloud 형태로 나타낼 수 있습니다.

<br>
<center><img src="../assets/img/autodrive/lidar/intro/11.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 지금까지 알아본 센서 중에 흔히 가장 많이 사용되고 다루어지는 센서는 카메라, 레이더, 라이다 센서입니다.
- 위 표와 같이 각 센서는 장단점의 영역이 다르게 분포되어 있음을 알 수 있습니다. 즉, 단일 센서로는 모든 영역을 다 커버할 수 없습니다. 따라서 **다양한 센서의 센서 퓨전**을 통하여 서로의 장단점을 보완하는 것이 추세입니다.
- 특히 정교한 3D 데이터를 측정하기 위해서는 라이다가 독보적인 것을 알 수 있습니다.

<br>

## **라이다의 활용 : 위치 추정, 객체 인지**

<br>


<br>