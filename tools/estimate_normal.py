import os
import sys
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(FILE_PATH, "..")
CONFIG_FILE = os.path.join(ROOT_DIR, "config/config.json")

config = json.load(open(CONFIG_FILE))
width = config["camera"]["width"]
height = config["camera"]["height"]
fx = config["camera"]["fx"]
fy = config["camera"]["fy"]
cx = width / 2
cy = height / 2

if len(sys.argv) != 2:
    print("Need target npy file to visualize")

npy_file = sys.argv[1]
depth = np.load(npy_file)

# points = np.zeros((width*height, 3), dtype=np.float32)
# count = 0
# for row in range(depth.shape[0]):
    # for col in range(depth.shape[1]):
        # z = depth[row][col]
        # x = (col - cx) * z / fx
        # y = (row - cy) * z / fy
        # points[count] = [x, y, z]
        # count += 1

# cloud = o3d.geometry.PointCloud()
# cloud.points = o3d.utility.Vector3dVector(points)

o3d_depth = o3d.geometry.Image(depth)
camera_model = o3d.camera.PinholeCameraIntrinsic()
camera_model.set_intrinsics(width, height, fx, fy, cx, cy)
cloud = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, intrinsic=camera_model, depth_scale=1.0)
print("created point cloud")
# o3d.visualization.draw_geometries([cloud])

param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=50)
cloud.estimate_normals(search_param=param)
print("estimated normals")

np_normals = np.asarray(cloud.normals)
normal_map = np.reshape(np_normals, [height, width, 3])
        
plt.imshow(normal_map)
plt.show()
