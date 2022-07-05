---
layout: post
title: 포인트 클라우드 처리를 위한 open3d 사용법 정리
date: 2021-06-30 00:00:00
img: autodrive/lidar/open3d/0.png
categories: [autodrive-lidar] 
tags: [라이다, open3d, 포인트 클라우드] # add tag
---

<br>

## **open3d로 point cloud 시각화 방법**

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