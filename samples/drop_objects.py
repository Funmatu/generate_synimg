import os
import sys

import time
import json
import random
import numpy as np

import cv2
import trimesh
import pybullet
import pybullet_data
import pyrender
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(FILE_PATH, "..")
CONFIG_FILE = os.path.join(ROOT_DIR, "config/config.json")
OBJ_MODEL_DIR = os.path.join(ROOT_DIR, "models/obj")
ENV_MODEL_DIR = os.path.join(ROOT_DIR, "models/env")
URDF_DIR = os.path.join(ROOT_DIR, "models/urdf")

# pybullet initialization
physicsClient = pybullet.connect(pybullet.GUI)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath()) 
pybullet.setGravity(0, 0, -10)

# Load plane
plane_mesh_dir = os.path.join(ENV_MODEL_DIR, "plane")
plane_trimesh = trimesh.load(plane_mesh_dir + "/" + "plane.obj")
plane_urdf_dir = os.path.join(URDF_DIR, "plane")
if not os.path.exists(plane_urdf_dir):
    os.makedirs(plane_urdf_dir)
    trimesh.exchange.urdf.export_urdf(plane_trimesh, plane_urdf_dir)
    print("Exported plane urdf")
planeId = pybullet.loadURDF(plane_urdf_dir + "/" + "plane.urdf")

# Load bin
bin_mesh_dir = os.path.join(ENV_MODEL_DIR, "bin")
bin_trimesh = trimesh.load(bin_mesh_dir + "/" + "bin.obj")
bin_urdf_dir = os.path.join(URDF_DIR, "bin")
if not os.path.exists(bin_urdf_dir):
    os.makedirs(bin_urdf_dir)
    trimesh.exchange.urdf.export_urdf(bin_trimesh, bin_urdf_dir)
    print("Exported bin urdf")
bin_z_origin = 0.012
binId = pybullet.loadURDF(bin_urdf_dir + "/" + "bin.urdf", [0.0, 0.0, bin_z_origin])

# Load obj
obj_name = sys.argv[1]
obj_mesh_dir = os.path.join(OBJ_MODEL_DIR, obj_name)
obj_trimesh = trimesh.load(obj_mesh_dir + "/" + obj_name + ".obj")
obj_urdf_dir = os.path.join(URDF_DIR, obj_name)
if not os.path.exists(obj_urdf_dir):
    os.makedirs(obj_urdf_dir)
    trimesh.exchange.urdf.export_urdf(obj_trimesh, obj_urdf_dir)
    print("Exported " + obj_name + " urdf")

obj_keys = []
num_objects = 7
for i in range(num_objects):
    drop_x = 0.0
    drop_y = 0.0
    drop_z = 1.0
    obj_id = pybullet.loadURDF(obj_urdf_dir + "/" + obj_name + ".urdf", [drop_x, drop_y, drop_z])
    obj_keys.append(obj_id)

    for k in range(200):
        pybullet.stepSimulation()
        time.sleep(3/240)

# breakpoint()
# Create scene for rendering
scene = pyrender.Scene()

# Add plane, bin, obj
plane_t, plane_q = pybullet.getBasePositionAndOrientation(planeId, physicsClient)
plane_R = Rotation.from_quat(plane_q).as_matrix()
plane_pose = np.eye(4)
plane_pose[:3, :3] = plane_R
plane_pose[:3, 3] = plane_t
plane_render_mesh = pyrender.Mesh.from_trimesh(plane_trimesh)
scene.add(plane_render_mesh, pose=plane_pose, name=planeId)

bin_t, bin_q = pybullet.getBasePositionAndOrientation(binId, physicsClient)
bin_R = Rotation.from_quat(bin_q).as_matrix()
bin_pose = np.eye(4)
bin_pose[:3, :3] = bin_R
bin_pose[:3, 3] = bin_t
bin_render_mesh = pyrender.Mesh.from_trimesh(bin_trimesh)
scene.add(bin_render_mesh, pose=bin_pose, name=binId)

for key in obj_keys:
    obj_t, obj_q = pybullet.getBasePositionAndOrientation(key, physicsClient)
    obj_R = Rotation.from_quat(obj_q).as_matrix()
    obj_pose = np.eye(4)
    obj_pose[:3, :3] = obj_R
    obj_pose[:3, 3] = obj_t
    obj_render_mesh = pyrender.Mesh.from_trimesh(obj_trimesh)
    scene.add(obj_render_mesh, pose=obj_pose, name=str(key))

# Add light
light = pyrender.SpotLight(color=np.ones(3), intensity=4.0)
light_pose = np.eye(4)
light_pose[2, 3] = 1.8
scene.add(light, pose=light_pose)

# Add camera
width = 2048
height = 1536
fx = 3700
fy = 3700
cx = width/2
cy = height/2
camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
camera_pose = np.eye(4)
camera_pose[2, 3] = 1.6
scene.add(camera, pose=camera_pose)

# breakpoint()
# Render scene
renderer = pyrender.OffscreenRenderer(width, height)
color, depth = renderer.render(scene)
plt.imshow(color)
plt.show()
plt.imshow(depth)
plt.show()

pybullet.disconnect()
