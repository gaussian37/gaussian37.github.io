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
- 아래 코드를 이용하면 어떤 축을 기준으로 grid를 생성합니다. `x_size`와 `y_size` 각각은 grid를 중점으로 부터 가로와 세로 방향으로 얼만큼 늘려갈 지에 해당합니다.
- `basis_axis`는 어떤 축을 기준으로 grid 평면을 만들 지 정하는 것 입니다. 예를 들어 basis_axis=2라고 하면 Z축을 0으로 둔 다음에 grid 평면을 만듭니다.

```python
def get_grid_lineset(x_size, y_size, basis_axis, grid_length, color):
    grid_vertexes_order = np.zeros((2*y_size+1, 2*x_size+1)).astype(np.int16)
    grid_vertexes = []
    vertex_order_index = 0
    for y in range(-y_size, y_size+1):
        for x in range(-x_size, x_size+1):
            grid_vertexes_order[y + y_size][x + x_size] = vertex_order_index
            if basis_axis == 0:
                grid_vertexes.append([0, grid_length*x, grid_length*y])
            elif basis_axis == 1:
                grid_vertexes.append([grid_length*x, 0, grid_length*y])
            elif basis_axis == 2:
                grid_vertexes.append([grid_length*x, grid_length*y, 0])
            else:
                pass                
            vertex_order_index += 1       
            
    next_y = [-1, 0, 0, 1]
    next_x = [0, -1, 1, 0]
    grid_lines = []
    for y in range(-y_size, y_size+1):
        for x in range(-x_size, x_size+1):
            here_y = y + y_size
            here_x = x + x_size
            for i in range(4):
                there_y = y + y_size + next_y[i]
                there_x = x + x_size + next_x[i]            
                if (0 <= there_y and there_y < 2*y_size+1) and (0 <= there_x and there_x < 2*x_size+1):
                    grid_lines.append([grid_vertexes_order[here_y][here_x],grid_vertexes_order[there_y][there_x]])      
                    
                    
    colors = [color for i in range(len(grid_lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(grid_vertexes),
        lines=o3d.utility.Vector2iVector(grid_lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set   
```