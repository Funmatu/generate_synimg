import os
import pybullet as p
import pybullet_data
import time
import math

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(FILE_PATH, "..")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

planeId = p.loadURDF("plane.urdf")
binId = p.loadURDF(MODEL_DIR + "/" + "env/bin/urdf/bin.urdf", [0.0, 0.0, 0.21])

objId1 = p.loadURDF(MODEL_DIR + "/" + "obj/choice/urdf/choice_vhacd.urdf", [0.1,0,2.0])
objId2 = p.loadURDF(MODEL_DIR + "/" + "obj/choice/urdf/choice_vhacd.urdf", [0,0.1,2.0])
objId3 = p.loadURDF(MODEL_DIR + "/" + "obj/choice/urdf/choice_vhacd.urdf", [-0.1,0,2.0])
objId4 = p.loadURDF(MODEL_DIR + "/" + "obj/choice/urdf/choice_vhacd.urdf", [0,-0.1,2.0])

p.setGravity(0,0,-10)

# Simulation for falling object to bin
for i in range(500):
    p.stepSimulation()
    time.sleep(1/240)

# Simulation for bin vibration
for h in range(10):
    for i in range(30):
        p.stepSimulation()    
        time.sleep(1/240)
        binPos, binOrn = p.getBasePositionAndOrientation(binId)
        p.applyExternalForce(binId, -1, [350.0, 0.0, 0.0], binPos, p.WORLD_FRAME)
    
    for i in range(30):
        p.stepSimulation()    
        time.sleep(1/240)
        binPos, binOrn = p.getBasePositionAndOrientation(binId)
        p.applyExternalForce(binId, -1, [-350.0, 0.0, 0.0], binPos, p.WORLD_FRAME)

print("Finish Simulation")
# time.sleep(20)
p.disconnect()
