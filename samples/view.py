import sys
import os
import trimesh
import pyrender


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(FILE_PATH, "..")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

cls = sys.argv[1]
obj_trimesh = trimesh.load(MODEL_DIR + '/' + f'obj/{cls}/{cls}.obj')
# breakpoint()
mesh = pyrender.Mesh.from_trimesh(obj_trimesh)
scene = pyrender.Scene()
scene.add(mesh)
pyrender.Viewer(scene, use_raymond_lighting=True)
