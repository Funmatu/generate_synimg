import os
import sys

import time
import json
import random
import datetime
import numpy as np

import cv2
import trimesh
import pybullet
import pybullet_data
import pyrender
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(FILE_PATH, "..")
CONFIG_FILE = os.path.join(ROOT_DIR, "config/config.json")
OBJ_MODEL_DIR = os.path.join(ROOT_DIR, "models/obj")
ENV_MODEL_DIR = os.path.join(ROOT_DIR, "models/env")
URDF_DIR = os.path.join(ROOT_DIR, "models/urdf")

args = sys.argv
cls = args[1]
env = args[2]
depth_out = args[3] if len(args)==4 else False

class ObjectState():
    def __init__(self, key, name, mesh, pose, isEnv):
        self.key = key
        self.name = name
        self.mesh = mesh
        self.pose = pose
        self.isEnv = isEnv

class GenerateSyntheticDataset():
    def __init__(self, config):
        self.config = config
        self.obj_list = []

        # Initialize renderer
        if self.config["pyrender"]["backend"] != "pygret":
            os.environ["PYOPENGL_PLATFORM"] = self.config["pyrender"]["backend"]
        width = self.config["camera"]["width"]
        height = self.config["camera"]["height"]
        self.renderer = pyrender.OffscreenRenderer(width, height)

        # Initialize physics simulation
        if self.config["pybullet"]["graphical"]:
            self.physicsClient = pybullet.connect(pybullet.GUI)
        else:
            self.physicsClient = pybullet.connect(pybullet.DIRECT)

        self.clsname = cls

    def __del__(self):
        pybullet.disconnect(self.physicsClient)

    def reset(self):
        self.obj_list = []
        self.num_objects = random.randint(self.config["num_objects_per_scene"][0], self.config["num_objects_per_scene"][1])
        # self.scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
        self.scene = pyrender.Scene()
        pybullet.resetSimulation(physicsClientId = self.physicsClient)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId = self.physicsClient) 
        pybullet.setGravity(0, 0, -10, physicsClientId = self.physicsClient)

    def load_mesh_urdf(self, name, isEnv = False):
        target_model_dir = os.path.join(ENV_MODEL_DIR, name) if isEnv else os.path.join(OBJ_MODEL_DIR, name)
        target_trimesh = trimesh.load_mesh(target_model_dir + "/" + name + ".obj")

        target_urdf_dir = os.path.join(URDF_DIR, name)
        if not os.path.exists(target_urdf_dir):
            os.makedirs(target_urdf_dir)
            trimesh.exchange.urdf.export_urdf(target_trimesh, target_urdf_dir)
            print("Exported " + name + " urdf")

        obj_pose = np.eye(4)
        if name == "bin":
            obj_pose[:3, 3] = [0.0, 0.0, 0.012]
        if not isEnv:
            obj_pose[:3, :3] = 2*np.pi * np.random.rand(3, 3) - np.pi
            obj_pose[:3, 3] = 0.2 * np.random.rand(3) - 0.1
            obj_pose[2, 3] = 1.0

        key = pybullet.loadURDF(target_urdf_dir + "/" + name + ".urdf", basePosition = obj_pose[:3, 3], physicsClientId = self.physicsClient)
        self.obj_list.append(ObjectState(key, name, target_trimesh, obj_pose, isEnv))

    def drop_objects(self, isLast = False):
        for i in range(200):
            pybullet.stepSimulation(physicsClientId = self.physicsClient)
            time.sleep(1/240)

        # Get final pose after simulation
        if isLast:
            for obj in self.obj_list:
                t, q = pybullet.getBasePositionAndOrientation(obj.key, physicsClientId = self.physicsClient)
                R = Rotation.from_quat(q).as_matrix()
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = t
                obj.pose = pose


    def in_workspace(self, position):
        if (self.config["workspace"]["x"][0] < position[0] < self.config["workspace"]["x"][1] and
            self.config["workspace"]["y"][0] < position[1] < self.config["workspace"]["y"][1] and
            self.config["workspace"]["z"][0] < position[2] < self.config["workspace"]["z"][1]):
            return True
        else:
            return False

    def align_objects(self, idx):
        row, col = 0, 0
        gap = 0.0005 * (random.randrange(3) + 1)
        
        surface_idx = idx % 6
        rotate = trimesh.transformations.rotation_matrix(np.radians(0), [1,0,0])
        if surface_idx == 0:
            pass
        elif surface_idx == 1:
            rotate = trimesh.transformations.rotation_matrix(np.radians(180), [1,0,0])
        elif surface_idx == 2:
            rotate = trimesh.transformations.rotation_matrix(np.radians(90), [1,0,0])
        elif surface_idx == 3:
            rotate = trimesh.transformations.rotation_matrix(np.radians(-90), [1,0,0])
        elif surface_idx == 4:
            rotate = trimesh.transformations.rotation_matrix(np.radians(90), [0,1,0])
        elif surface_idx == 5:
            rotate = trimesh.transformations.rotation_matrix(np.radians(-90), [0,1,0])

        surface_idx2 = idx % 4
        rotate2 = trimesh.transformations.rotation_matrix(np.radians(0), [0,0,1])
        if surface_idx2 == 0:
            pass
        elif surface_idx2 == 1:
            rotate2 = trimesh.transformations.rotation_matrix(np.radians(90), [0,0,1])
        elif surface_idx2 == 2:
            rotate2 = trimesh.transformations.rotation_matrix(np.radians(-90), [0,0,1])
        elif surface_idx2 == 3:
            rotate2 = trimesh.transformations.rotation_matrix(np.radians(180), [0,0,1])

        colup_direction = random.randrange(2)
        rowup_direction = random.randrange(2)
        obj_keys_str = []
        for obj in self.obj_list:
            if obj.isEnv: continue

            obj_trimesh = obj.mesh
            obj_trimesh.apply_transform(rotate)
            obj_trimesh.apply_transform(rotate2)
            
            surface_idx3 = random.randrange(4)
            rotate3 = trimesh.transformations.rotation_matrix(np.radians(0), [0,0,1])
            if surface_idx3 == 0:
                pass
            elif surface_idx3 == 1:
                rotate3 = trimesh.transformations.rotation_matrix(np.radians(180), [0,0,1])
            elif surface_idx3 == 2:
                rotate3 = trimesh.transformations.rotation_matrix(np.radians(180), [0,1,0])
            elif surface_idx3 == 3:
                rotate3 = trimesh.transformations.rotation_matrix(np.radians(180), [1,0,0])
            obj_trimesh.apply_transform(rotate3)

            bbox = obj_trimesh.bounds
            obj_xlength, obj_ylength, obj_zlength = 2 * abs(bbox[0])
            
            # initial_position = np.array([0, 0, 0.012 + obj_zlength])
            initial_position = np.array([self.config["workspace"]["x"][0] + obj_xlength/2, 
                                         self.config["workspace"]["y"][0] + obj_ylength/2,
                                         0.012 + obj_zlength
                               ])
            initial_pose = np.eye(3)

            lineup_position = initial_position.copy()
            # collision_position = initial_position.copy()
            
            lineup_position[0] += obj_xlength * col + gap * col
            lineup_position[1] += obj_ylength * row + gap * row

            collision_position = lineup_position.copy()
            collision_position[0] += obj_xlength/2
            collision_position[1] += obj_ylength/2
            # if rowup_direction == 0:
                # # lineup from top
                # if row % 2 != 0:
                    # lineup_position[1] += obj_ylength * (row//2+1) + gap * (row//2)
                    # collision_position[1] = lineup_position[1] + obj_ylength/2
                # else:
                    # lineup_position[1] += obj_ylength * (row//2) * -1 + gap * (row//2) * -1
                    # collision_position[1] = lineup_position[1] - obj_ylength/2
            # elif rowup_direction == 1:
                # # lineup from bottom
                # if row % 2 == 0:
                    # lineup_position[1] += obj_ylength * (row//2) + gap * (row//2)
                    # collision_position[1] = lineup_position[1] + obj_ylength/2
                # else:
                    # lineup_position[1] += obj_ylength * (row//2+1) * -1 + gap * (row//2) * -1
                    # collision_position[1] = lineup_position[1] - obj_ylength/2
            
            # if colup_direction == 0:
                # # lineup from right
                # if col % 2 != 0:
                    # lineup_position[0] += obj_xlength * (col//2 + 1) + gap * (col//2 + 1)
                    # collision_position[0] = lineup_position[0] + obj_xlength/2
                # else:
                    # lineup_position[0] += obj_xlength * (col//2) * -1 + gap * (col//2) * -1
                    # collision_position[0] = lineup_position[0] - obj_xlength/2
            # elif colup_direction == 1:
                # # lineup from left
                # if col % 2 == 0:
                    # lineup_position[0] += obj_xlength * (col//2) + gap * (col//2)
                    # collision_position[0] = lineup_position[0] + obj_xlength/2
                # else:
                    # lineup_position[0] += obj_xlength * (col//2 + 1) * -1 + gap * (col//2 + 1) * -1
                    # collision_position[0] = lineup_position[0] - obj_xlength/2

            obj_pose = np.eye(4)
            obj_pose[:3, :3] = initial_pose
            obj_pose[:3, 3] = lineup_position
            row += 1
            if not self.in_workspace(collision_position) or row == 8:
                col += 1
                next_col_position = initial_position.copy()
                next_col_position[0] += obj_xlength * col + gap * col

                # collision_position = initial_position.copy()
                collision_position = next_col_position.copy()
                collision_position[0] += obj_xlength/2
                collision_position[1] += obj_ylength/2

                # if colup_direction == 0:
                    # if col % 2 != 0:
                        # next_col_position[0] += obj_xlength * (col//2 + 1) + gap * (col//2 + 1)
                        # collision_position[0] = next_col_position[0] + obj_xlength/2
                    # else:
                        # next_col_position[0] += obj_xlength * (col//2) * -1 + gap * (col//2) * -1
                        # collision_position[0] = next_col_position[0] - obj_xlength/2

                # elif colup_direction == 1:
                    # if col % 2 == 0:
                        # next_col_position[0] += obj_xlength * (col//2) + gap * (col//2)
                        # collision_position[0] = next_col_position[0] + obj_xlength/2
                    # else:
                        # next_col_position[0] += obj_xlength * (col//2 + 1) * -1 + gap * (col//2 + 1) * -1
                        # collision_position[0] = next_col_position[0] - obj_xlength/2

                if not self.in_workspace(collision_position):
                    col -= 1
                    break
                    # continue

                obj_pose[:3, 3] = next_col_position
                row = 1

            if random.randrange(10) == 0:
                continue
            obj.pose = obj_pose

    def translate_objects(self, idx):
        workspace_xlength = self.config["workspace"]["x"][1] - self.config["workspace"]["x"][0]
        workspace_ylength = self.config["workspace"]["y"][1] - self.config["workspace"]["y"][0]
        workspace_zlength = self.config["workspace"]["z"][1] - self.config["workspace"]["z"][0]

        translation_idx = idx//6
        if translation_idx > 2:
            translation_idx = random.randint(0, 2)

        min_x, max_x, min_y, max_y = (0, 0, 0, 0)
        for obj in self.obj_list:
            if obj.isEnv: continue

            if obj.pose[0, 3] < min_x:
                min_x = obj.pose[0, 3]
            if obj.pose[0, 3] > max_x:
                max_x = obj.pose[0, 3]
            if obj.pose[1, 3] < min_y:
                min_y = obj.pose[1, 3]
            if obj.pose[1, 3] > max_y:
                max_y = obj.pose[1, 3]

        for obj in self.obj_list:
            if obj.isEnv: continue

            bbox = obj.mesh.bounds
            obj_xlength, obj_ylength, obj_zlength = 2 * abs(bbox[0])

            if translation_idx == 0:
                if idx % 2 == 0:
                    obj.pose[0, 3] -= min_x - self.config["workspace"]["x"][0] - obj_xlength/2
                    obj.pose[1, 3] += self.config["workspace"]["y"][1] - max_y - obj_ylength/2
                if idx % 2 == 1:
                    obj.pose[1, 3] += self.config["workspace"]["y"][1] - max_y - obj_ylength/2
            elif translation_idx == 1:
                if idx % 2 == 0:
                    obj.pose[0, 3] += self.config["workspace"]["x"][1] - max_x - obj_xlength/2
                    obj.pose[1, 3] += self.config["workspace"]["y"][1] - max_y - obj_ylength/2
                if idx % 2 == 1:
                    obj.pose[0, 3] += self.config["workspace"]["x"][1] - max_x - obj_xlength/2
                    obj.pose[1, 3] -= min_y - self.config["workspace"]["y"][0] - obj_ylength/2
            elif translation_idx == 2:
                if idx % 2 == 0:
                    obj.pose[1, 3] -= min_y - self.config["workspace"]["y"][0] - obj_ylength/2
                if idx % 2 == 1:
                    obj.pose[0, 3] -= min_x - self.config["workspace"]["x"][0] - obj_xlength/2
                    obj.pose[1, 3] -= min_y - self.config["workspace"]["y"][0] - obj_ylength/2


    def add_object(self, augment_param = None):
        for obj in self.obj_list[:]:
            target_model_dir = os.path.join(ENV_MODEL_DIR, obj.name) if obj.isEnv else os.path.join(OBJ_MODEL_DIR, obj.name)
            texture = cv2.cvtColor(cv2.imread(target_model_dir + "/" + obj.name + ".png"), cv2.COLOR_BGR2RGB)
            texture = pyrender.Texture(source=texture, source_channels='RGB')
            if augment_param:
                texture = augment_param.augment_image(texture)

            material = pyrender.MetallicRoughnessMaterial(baseColorTexture=texture, metallicFactor=0.2, roughnessFactor=0.8)
            obj_render_mesh = pyrender.Mesh.from_trimesh(obj.mesh, material=material)
            
            if self.in_workspace(obj.pose[:3, 3]) or obj.isEnv:
                self.scene.add(obj_render_mesh, pose=obj.pose, name=str(obj.key))
            else:
                self.obj_list.remove(obj)

    def add_light(self, light_color=np.ones(3)):
        # light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
        # light_pose = np.eye(4)
        # self.scene.add(light, pose=light_pose)
        
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)
        
            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)
        
            matrix = np.eye(4)
            matrix[:3,:3] = np.c_[x,y,z]
            node = pyrender.Node(
                light=pyrender.DirectionalLight(color=light_color, intensity=1.0),
                matrix=matrix
            )
            self.scene.add_node(node)

    def add_camera(self, camera_pose = np.eye(4)):
        fx = self.config["camera"]["fx"]
        fy = self.config["camera"]["fy"]
        cx = self.config["camera"]["width"] / 2
        cy = self.config["camera"]["height"] /2
        camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
        self.scene.add(camera, pose=camera_pose)

    def rendering(self):
        self.color, self.depth = self.renderer.render(self.scene)
        if self.config["pyrender"]["view"]:
            plt.imshow(color)
            plt.show()

        return self.color, self.depth

    def get_segmented_info(self, scene_type):
        width = self.config["camera"]["width"]
        height = self.config["camera"]["height"]
        full_depth = self.depth.copy()
        modal_data = np.zeros((full_depth.shape[0], full_depth.shape[1], self.num_objects), dtype=np.uint8)
        amodal_data = np.zeros((full_depth.shape[0], full_depth.shape[1], self.num_objects), dtype=np.uint8)
        
        flags = pyrender.RenderFlags.DEPTH_ONLY
        
        obj_mesh_nodes = [next(iter(self.scene.get_nodes(name=str(obj.key)))) for obj in self.obj_list if not obj.isEnv]
        for mn in self.scene.mesh_nodes:
            mn.mesh.is_visible = False
        
        for j, node in enumerate(obj_mesh_nodes):
            node.mesh.is_visible = True
        
            depth = self.renderer.render(self.scene, flags=flags)
            amodal_mask = depth > 0.0
            modal_mask = np.logical_and(
                (np.abs(depth - full_depth) < 1e-6), full_depth > 0.0
            )
            amodal_data[amodal_mask, j] = np.iinfo(np.uint8).max
            modal_data[modal_mask, j] = np.iinfo(np.uint8).max
            node.mesh.is_visible = False
        
        for mn in self.scene.mesh_nodes:
            mn.mesh.is_visible = True
        
        amodal_segmasks, modal_segmasks = amodal_data, modal_data
        
        modal_segmask_arr = np.iinfo(np.uint8).max * np.ones([height, width, self.num_objects], dtype=np.uint8)
        amodal_segmask_arr = np.iinfo(np.uint8).max * np.ones([height, width, self.num_objects], dtype=np.uint8)
        stacked_segmask_arr = np.zeros([height, width, 1], dtype=np.uint8)
        
        modal_segmask_arr[:,:,:self.num_objects] = modal_segmasks
        amodal_segmask_arr[:,:,:self.num_objects] = amodal_segmasks
        for j in range(self.num_objects):
            this_obj_px = np.where(modal_segmasks[:,:,j] > 0)
            stacked_segmask_arr[this_obj_px[0], this_obj_px[1], 0] = j + 1
        
        if self.config["pyrender"]["view"]:
            plt.imshow(stacked_segmask_arr.squeeze())
            plt.show()

        regions = []
        for j in range(self.num_objects):
            img_modal = modal_segmask_arr[:,:,j]
            modal_pixel_num = np.count_nonzero(img_modal>125)
            img_amodal = amodal_segmask_arr[:,:,j]
            amodal_pixel_num = np.count_nonzero(img_amodal>125)
        
            if (scene_type == "aligned") or (scene_type == "clutter" and modal_pixel_num > amodal_pixel_num * 0.6):
            # if modal_pixel_num > amodal_pixel_num * 0.6:
                attr = {}
                attr["region_attributes"] = {"name": self.clsname}
                ret, thresh = cv2.threshold(img_modal, 127, 255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if (len(contours) == 0) : continue
        
                contours_index = 0
                # if len(countours) > 1 then adopt big contour
                for c in range(len(contours)):
                    if c != len(contours) - 1:
                        if len(contours[c]) < len(contours[c + 1]):
                            contours_index = c + 1
        
                contours = contours[contours_index]
                all_points_x = [int(contours[x][0][0]) for x in range(len(contours))]
                all_points_y = [int(contours[y][0][1]) for y in range(len(contours))]
                attr["shape_attributes"] = {"name": "polyline", "all_points_x": all_points_x, "all_points_y": all_points_y}
                regions.append(attr)

        return regions

    def generate(self, dataset_type, scene_type, cnt=0):
        now = datetime.datetime.now()
        dt_folder = str(now.year) + str(now.month).zfill(2) + str(now.day).zfill(2)
        image_save_dir = os.path.join(ROOT_DIR, self.config["image_save_dir"] + "/" + dt_folder)

        for i, target_object in enumerate(self.config["target_objects"]):
            self.target_object = target_object
            print(" Target Object is '" + self.target_object + "'")

            self.obj_image_save_dir = os.path.join(image_save_dir, dataset_type + "/" + self.clsname + "/")
            if not os.path.exists(self.obj_image_save_dir):
                os.makedirs(self.obj_image_save_dir)

            count = i * (self.config['num_generated_images']*3) + cnt
            dataset = {}
            for camera_height in [1.4, 1.6, 1.8]:
                N1 = 4 if dataset_type=='val' else 1 # weight of train vs val = 4 : 1
                N2 = 5 if scene_type=='clutter' else 1 # weight of aligned vs clutter = 5 : 1
                N = N1 * N2
                for i in range(self.config["num_generated_images"]//N):
                    self.reset()
                    # print("  Step " + str(i))
                    
                    self.load_mesh_urdf(env, isEnv=True)
                    # self.load_mesh_urdf("plane", isEnv = True)
                    # self.load_mesh_urdf("bin", isEnv = True)

                    for j in range(self.num_objects):
                        self.load_mesh_urdf(self.target_object)
                        if scene_type == "clutter":
                            self.drop_objects(isLast = j == self.num_objects - 1)

                    if scene_type == "aligned":
                        self.align_objects(idx = i)
                        self.translate_objects(idx = i)

                    augment_param = iaa.WithHueAndSaturation([
                                  iaa.WithChannels(0, iaa.Add((-5, 5))),
                                  iaa.WithChannels(1, [
                                      iaa.Multiply((0.8, 1.1)),
                                      iaa.LinearContrast((0.8, 1.1))
                                  ])
                    ])
                    light_color = (1.0 - 0.1) * np.random.rand(3) + 0.1

                    self.add_object()
                    self.add_light(light_color)
                    self.add_camera()

                    camera_nodes = self.scene.camera_nodes
                    self.scene.remove_node(next(iter(camera_nodes)))
                    camera_pose = np.eye(4)
                    camera_pose[2, 3] = camera_height
                    self.add_camera(camera_pose)

                    color, depth = self.rendering()
                    filename = str(count).zfill(4) + '.png'
                    cv2.imwrite(self.obj_image_save_dir + "/" + filename, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
                    dataset[filename] = {"filename": filename, "file_attributes":{}, "size":0}
                    
                    # depth unit is [m]
                    if depth_out:
                        np.save(self.obj_image_save_dir + "/" + str(count).zfill(4) + "_depth", depth)

                    regions = self.get_segmented_info(scene_type)
                    dataset[filename]["regions"] = regions
                    
                    count += 1

            with open(self.obj_image_save_dir+"/label.json", "a") as f:
                json.dump(dataset, f)
        return count

def main():
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    gsd = GenerateSyntheticDataset(config)
    print("Create aligned train dataset")
    cnt = gsd.generate("train", "aligned")
    # print("Create clutter train dataset")
    # gsd.generate("train", "clutter", cnt)
    print("create aligned val dataset")
    cnt = gsd.generate("val", "aligned")
    # print("create clutter val dataset")
    # gsd.generate("val", "clutter", cnt)
    
    del gsd

if __name__ == "__main__":
    print(f'Start: {datetime.datetime.now()}')
    main()
    print(f'Finished: {datetime.datetime.now()}')
