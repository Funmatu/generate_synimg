import os
import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(FILE_PATH, "..")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

obj_trimesh = trimesh.load(MODEL_DIR + '/' + 'obj/choice/choice.obj')
mesh = pyrender.Mesh.from_trimesh(obj_trimesh)
scene = pyrender.Scene()
scene.add(mesh)
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
s = np.sqrt(2)/2
camera_pose = np.array([
   [0.0, -s,   s,   0.3],
   [1.0,  0.0, 0.0, 0.0],
   [0.0,  s,   s,   0.35],
   [0.0,  0.0, 0.0, 1.0],
])
scene.add(camera, pose=camera_pose)
light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                           innerConeAngle=np.pi/16.0,
                           outerConeAngle=np.pi/6.0)
scene.add(light, pose=camera_pose)
r = pyrender.OffscreenRenderer(400, 400)
color, depth = r.render(scene)
plt.figure()
plt.subplot(1,2,1)
plt.axis('off')
plt.imshow(color)
plt.subplot(1,2,2)
plt.axis('off')
plt.imshow(depth, cmap=plt.cm.gray_r)
plt.show()
