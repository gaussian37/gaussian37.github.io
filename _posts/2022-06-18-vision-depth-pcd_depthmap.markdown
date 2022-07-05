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

- ### [PCD to Depth Map 원리](#)
- ### [PCD to Depth Map 실습](#)
- ### [Depth Map to PCD 원리](#)
- ### [Depth Map to PCD 실습](#)

<br>

## **PCD to Depth Map 원리**

<br>



<br>

## **PCD to Depth Map 실습**

<br>

- 지금부터 앞에서 다룬 개념을 이용하여 `PCD`를 `Depth Map`으로 바꾸는 방법에 대하여 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/depth/pcd_depthmap/1.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 실습을 위해 사용하는 센서셋은 `KITTI` 데이터의 센서셋 구조를 따라서 해볼 예정입니다. 라이다와 카메라의 장착 위치 및 좌표계를 확인하시면 아래 코드를 이해하는 데 도움이 됩니다.

<br>
<center><img src="../assets/img/vision/depth/pcd_depthmap/2.png" alt="Drawing" style="width: 1200px;"/></center>
<br>

- 위 내용은 `KITTI`의 캘리브레이션 관련 파라미터를 정리해 놓은 양식입니다. 실습에서 사용할 양식도 위 구조와 동일합니다.
- `P0` ~ `P3`는 카메라 intrinsic 파라미터이고 동차 좌표계 기준으로 12개의 값을 가집니다.
- `R0_rect`는 왜곡 (distortion)을 제거하기 위한 카메라 좌표계에서 왜곡이 제거된 카메라 좌표계로 변환하기 위해 사용합니다. KITTI에서는 스테레오 카메라를 사용하였고 이 카메라에서 발생되는 왜곡을 보상하기 위해 사용되는 값입니다.
- `Tr_velo_to_cam`과 `Tr_imu_to_velo`는 A 센서 → B 센서로 좌표축을 옮기기 위한 extrinsic 파라미터 입니다.

<br>
<center><img src="../assets/img/vision/depth/pcd_depthmap/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 포인트 클라우드의 뎁스 정보는 위 색상과 같이 0 ~ 50m 범위에 대하여 색을 나타내 보겠습니다. 코드에서 어떤점을 수정하면 범위를 늘릴 수 있는 지 이후에 설명 드리겠습니다.

<br>
<center><img src="../assets/img/vision/depth/pcd_depthmap/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 최종적으로 위의 첫번째 이미지에 두번째 이미지와 같이 포인트 클라우드를 색상을 이용하여 표현하는 것을 목적으로 합니다.

<br>

```python
class LiDAR2Camera(object):
    def __init__(self, calib_file, depth_column=0):
        calibs = self.read_calib_file(calib_file)
        
        P = calibs["P2"]
        self.P = np.reshape(P, [3, 4])
        
        # Rigid transform from Velodyne coord to reference camera coord
        V2C = calibs["Tr_velo_to_cam"]
        self.V2C = np.reshape(V2C, [3, 4])
        # Rotation from reference camera coord to rect camera coord
        R0 = calibs["R0_rect"]
        self.R0 = np.reshape(R0, [3, 3])
        self.depth_column = depth_column
        
        self.imgfov_pts_2d = None
        self.imgfov_pc_velo = None
        self.imgfov_depth = None

    def read_calib_file(self, filepath):
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data
    
    def cart2hom(self, pts_3d):
        """ Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom
    
    def project_velo_to_image(self, pts_3d_velo, debug=True):
        '''
        Input: 3D points in Velodyne Frame [nx3]
        Output: 2D Pixels in Image Frame [nx2]
        '''

        R0_homo = np.vstack([self.R0, [0, 0, 0]])
        R0_homo_2 = np.column_stack([R0_homo, [0, 0, 0, 1]])

        # PxR0
        p_r0 = np.dot(self.P, R0_homo_2) 

        # PxROxRT
        p_r0_rt =  np.dot(p_r0, np.vstack((self.V2C, [0, 0, 0, 1]))) 
        pts_3d_homo = np.column_stack([pts_3d_velo, np.ones((pts_3d_velo.shape[0],1))])

        # P x RO x RT x X
        p_r0_rt_x = np.dot(p_r0_rt, np.transpose(pts_3d_homo))
        pts_2d = np.transpose(p_r0_rt_x)

        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]

        if debug == True:
            print("R0_homo_2 : ", R0_homo_2)
            print("")
            print("p_r0 : ", p_r0)
            print("")
            print("rt_homo : ", np.vstack((self.V2C, [0, 0, 0, 1])))
            print("")
            print("p_r0_rt : ", p_r0_rt)
            print("")
            print("pts_3d_homo : ", pts_3d_homo)
            print("")
            print("p_r0_rt_x : ", p_r0_rt_x)
            print("")
            print("pts_2d : ", pts_2d)
            print("")

        return pts_2d[:, 0:2]
    
    def get_lidar_in_image_fov(self, pc_velo, xmin, ymin, xmax, ymax, clip_distance=0.1):
        """ Filter lidar points, keep those in image FOV """

        # point cloud → points in 2d image
        pts_2d = self.project_velo_to_image(pc_velo)

        # points index in fov
        fov_inds = (
            (pts_2d[:, 0] < xmax)
            & (pts_2d[:, 0] >= xmin)
            & (pts_2d[:, 1] < ymax)
            & (pts_2d[:, 1] >= ymin)
        )

        # ############# check lidar axis ###############
        # depth orientation of point cloud is x (pc_velo[:, 0])
        # We don't want things that are closer to the clip distance (2m)
        fov_inds = fov_inds & (pc_velo[:, self.depth_column] > clip_distance)
        imgfov_pc_velo = pc_velo[fov_inds, :]
        pts_2d = pts_2d[fov_inds, :]

        return imgfov_pc_velo, pts_2d
    
    def get_min_dist_lidar_in_image_fov(self, imgfov_pc_velo, imgfov_pts_2d):
        
        imgfov_pts_2d = np.round(imgfov_pts_2d)
        df = pd.DataFrame({
                'width' : imgfov_pts_2d[:, 0],
                'height' : imgfov_pts_2d[:, 1],
                'X' : imgfov_pc_velo[:, 0],
                'Y' : imgfov_pc_velo[:, 1],
                'Z' : imgfov_pc_velo[:, 2]
        })
        
        # depth axis on lidar is X
        if self.depth_column == 0:
            min_depth_df = df.groupby(['width', 'height', 'Y', 'Z'], as_index=False).min()      
        # depth axis on lidar is Y
        elif self.depth_column == 1:
            min_depth_df = df.groupby(['width', 'height', 'X', 'Z'], as_index=False).min()
        # depth axis on lidar is Z
        elif self.depth_column == 2:
            min_depth_df = df.groupby(['width', 'height', 'X', 'Y'], as_index=False).min()
        else:
            pass                
        
        min_depth_np = np.array(min_depth_df)     
        imgfov_pts_2d = np.c_[min_depth_df['width'].to_numpy(), min_depth_df['height'].to_numpy()]
        imgfov_pc_velo = np.c_[min_depth_df['X'].to_numpy(), min_depth_df['Y'].to_numpy(), min_depth_df['Z'].to_numpy()]
        
        return imgfov_pc_velo, imgfov_pts_2d
        

    def show_lidar_on_image(self, pc_velo, img, range_meter=50.0, min_depth_filter=True, debug=False):

        """ Project LiDAR points to image """
        imgfov_pc_velo, imgfov_pts_2d = self.get_lidar_in_image_fov(
            pc_velo, 0, 0, img.shape[1], img.shape[0], debug
        )
        
        if min_depth_filter:
            imgfov_pc_velo, imgfov_pts_2d = self.get_min_dist_lidar_in_image_fov(imgfov_pc_velo, imgfov_pts_2d)
            
        self.imgfov_pts_2d = imgfov_pts_2d
        self.imgfov_pc_velo = imgfov_pc_velo
        self.imgfov_depth = imgfov_pc_velo[:, self.depth_column]
        
        cmap = plt.cm.get_cmap("jet", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        
        for i in range(self.imgfov_pts_2d.shape[0]):
            #print(depth)
            # ############# check lidar axis ###############
            # depth orientation of point cloud is x (pc_velo[:, 0])
            depth = self.imgfov_depth[i]
            color_index = int(255 * min(depth,range_meter)/range_meter)
            color = cmap[color_index, :]
            cv2.circle(
                img,(int(np.round(self.imgfov_pts_2d[i, 0])), int(np.round(self.imgfov_pts_2d[i, 1]))), 2,
                color=tuple(color),
                thickness=-1,
            )
        return img
```

<br>


<br>

## **Depth Map to PCD 원리**

<br>



<br>

## **Depth Map to PCD 실습**

<br>



<br>
