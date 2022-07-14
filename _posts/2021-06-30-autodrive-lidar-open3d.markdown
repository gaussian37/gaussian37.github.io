---
layout: post
title: 포인트 클라우드 처리를 위한 open3d 사용법 정리
date: 2021-06-30 00:00:00
img: autodrive/lidar/open3d/0.png
categories: [autodrive-lidar] 
tags: [라이다, open3d, 포인트 클라우드] # add tag
---

<br>

## **목차**

<br>

- ### **open3d로 point cloud 시각화 방법**
- ### **open3d로 grid 생성하는 방법**

<br>

## **open3d로 point cloud 시각화 방법**

<br>

- 아래 코드는 PointCloud 또는 Numpy 형태의 포인트 클라우드를 입력받아서 시각화 하는 기본적인 코드입니다.
- 추가적으로 좌표축도 같이 표현하고 있으며 R, G, B 색 순서로 X, Y, Z 축을 나타냅니다.

<br>

```python
def show_open3d_pcd(pcd, show_origin=True, origin_size=3, show_grid=True):
    cloud = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector
    
    if isinstance(pcd, type(cloud)):
        pass
    elif isinstance(pcd, np.ndarray):
        cloud.points = v3d(pcd)
        
    coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=origin_size, origin=np.array([0.0, 0.0, 0.0]))
    
    # set front, lookat, up, zoom to change initial view
    o3d.visualization.draw_geometries([cloud, coord])
```

<br>

## **open3d로 grid 생성하는 방법**

<br>

- open3d로 포인트 클라우드를 읽었을 때, 그 포인트 클라우드의 대략적인 위치를 파악하기가 어렵습니다. 따라서 grid 형태로 거리를 참조할 수 있는 정보가 필요합니다.
- 아래 코드를 이용하면 XY, YZ, ZX 방향으로 grid를 그려서 포인트 클라우드의 위치를 파악하기 용이합니다.

```python
import open3d as o3d
import numpy as np

def get_grid_lineset(h_min_val, h_max_val, w_min_val, w_max_val, ignore_axis, grid_length, color):
    
    num_h_grid = int(np.round((h_max_val - h_min_val) // grid_length, -1))
    num_w_grid = int(np.round((w_max_val - w_min_val) // grid_length, -1))
    
    num_h_grid_mid = num_h_grid // 2
    num_w_grid_mid = num_w_grid // 2
    
    grid_vertexes_order = np.zeros((num_h_grid, num_w_grid)).astype(np.int16)
    grid_vertexes = []
    vertex_order_index = 0
    
    for h in range(num_h_grid):
        for w in range(num_w_grid):
            grid_vertexes_order[h][w] = vertex_order_index
            if ignore_axis == 0:
                grid_vertexes.append([0, grid_length*w + w_min_val, grid_length*h + h_min_val])
            elif ignore_axis == 1:
                grid_vertexes.append([grid_length*h + h_min_val, 0, grid_length*w + w_min_val])
            elif ignore_axis == 2:
                grid_vertexes.append([grid_length*w + w_min_val, grid_length*h + h_min_val, 0])
            else:
                pass                
            vertex_order_index += 1       
            
    next_h = [-1, 0, 0, 1]
    next_w = [0, -1, 1, 0]
    grid_lines = []
    for h in range(num_h_grid):
        for w in range(num_w_grid):
            here_h = h
            here_w = w
            for i in range(4):
                there_h = h + next_h[i]
                there_w = w +  next_w[i]            
                if (0 <= there_h and there_h < num_h_grid) and (0 <= there_w and there_w < num_w_grid):
                    grid_lines.append([grid_vertexes_order[here_h][here_w], grid_vertexes_order[there_h][there_w]])
                    
                    
    colors = [color for i in range(len(grid_lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(grid_vertexes),
        lines=o3d.utility.Vector2iVector(grid_lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set   

def show_open3d_pcd(raw, show_origin=True, origin_size=3, 
                    show_grid=True, grid_len=1, 
                    voxel_size=0, 
                    range_min_xyz=(-80, -80, 0), range_max_xyz=(80, 80, 80)):
    '''
    - raw : numpy 2d array (size : (n, 3)) or o3d.geometry.PointCloud
    - show_origin : show origin XYZ coordinate. (X=red, Y=green, Z=Blue)
    - origin_size : size of origin coordinate.
    - show_grid : if true, show grid in xy, yz, zx plane with 'grid_len' length (default : gray line) and 5 times of 'grid_len' (default : red line)
    - voxel_size : voxel size to downsampling
    - range_min_xyz : grid min range of xyz orientation
    - range_max_xyz : grid max range of xyz orientation

    '''
    pcd = o3d.geometry.PointCloud()    
    
    if isinstance(raw, type(pcd)):
        pass
    elif isinstance(raw, np.ndarray):
        pcd.points = o3d.utility.Vector3dVector(raw)        
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        
    pcd_point = np.array(pcd.points)
    inrange_inds = (pcd_point[:, 0] > range_min_xyz[0]) & \
                    (pcd_point[:, 1] > range_min_xyz[1]) & \
                    (pcd_point[:, 2] > range_min_xyz[2]) & \
                    (pcd_point[:, 0] < range_max_xyz[0]) & \
                    (pcd_point[:, 1] < range_max_xyz[1]) & \
                    (pcd_point[:, 2] < range_max_xyz[2])    
    
    pcd_point = pcd_point[inrange_inds]
    filtered_raw = pcd_point
    pcd.points = o3d.utility.Vector3dVector(filtered_raw)
        
    x_min_val, y_min_val, z_min_val = range_min_xyz
    x_max_val, y_max_val, z_max_val = range_max_xyz
    
    coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=origin_size, origin=np.array([0.0, 0.0, 0.0]))
    
    R, G, B = 0.8, 0.8, 0.8
    lineset_yz = get_grid_lineset(z_min_val, z_max_val, y_min_val, y_max_val, 0, grid_len, [R, G, B])
    lineset_zx = get_grid_lineset(x_min_val, x_max_val, z_min_val, z_max_val, 1, grid_len, [R, G, B])
    lineset_xy = get_grid_lineset(y_min_val, y_max_val, x_min_val, x_max_val, 2, grid_len, [R, G, B]) 
    
    R, G, B = 1, 0, 0
    lineset_yz_5 = get_grid_lineset(z_min_val, z_max_val, y_min_val, y_max_val, 0, grid_len*5, [R, G, B])
    lineset_zx_5 = get_grid_lineset(x_min_val, x_max_val, z_min_val, z_max_val, 1, grid_len*5, [R, G, B])
    lineset_xy_5 = get_grid_lineset(y_min_val, y_max_val, x_min_val, x_max_val, 2, grid_len*5, [R, G, B]) 
    
    # set front, lookat, up, zoom to change initial view
    o3d.visualization.draw_geometries([pcd, coord,
                                       lineset_yz_5, lineset_zx_5, lineset_xy_5,
                                       lineset_xy, lineset_yz, lineset_zx
                                      ])  
```

<br>

- 아래와 같이 호출하여 사용할 수 있습니다. 아래 `bin` 파일은 `KITTI` 데이터셋의 포인트 클라우드 데이터이며 `bin` 형태로 저장되어 있습니다. 아래 파일을 읽으면 열 방향으로 `(X, Y, Z, Intensity)` 순서로 읽을 수 있으며 (N, 4)의 크기 행렬을 가지게 됩니다.

<br>

```python
raw = np.fromfile("00000001.bin", dtype=np.float32).reshape((-1, 4))
raw = raw[:, :3]
show_open3d_pcd(pcd, origin_size=3, range_min_xyz=(-40, -40, -5), range_max_xyz=(40, 40, 5))
```

<br>

<br>
<center><img src="../assets/img/autodrive/lidar/open3d/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림에서 연한 회색 선은 기본 값인 1m를 나타내고 빨간색 선은 기본 값의 5배인 5m를 나타냅니다. 위 코드에서 기본값을 2m로 바꾸면 빨간색 선은 5배인 10m를 나타냅니다.

<br>
<center><img src="../assets/img/autodrive/lidar/open3d/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- `show_open3d_pcd` 함수에서 `origin_size=3`으로 지정하였고 origin이 3m 크기로 나타난 것을 확인할 수 있습니다.

<br>
<center><img src="../assets/img/autodrive/lidar/open3d/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- `range_min_xyz=(-40, -40, -5)`, `range_max_xyz=(40, 40, 5)` 의 옵션에 따라 x, y, z 방향으로 포인트들을 필터링하고 남겨진 포인트들을 적당한 마진을 추가하여 XYZ 평면에 표시한 형태입니다.



