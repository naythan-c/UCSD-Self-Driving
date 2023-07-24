import os, inspect
import pybullet as p
import pybullet_data
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import random
import open3d as o3d
from PIL import Image, ImageOps
import cv2


def depth_to_pcd(depth, fx, fy, cx, cy):
    """
    depth: depth image
    fx, fy, cx, cy: camera intrinsics
    """
    # Convert to Open3D depth image
    depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))  # Open3D expects depth in mm

    # Define camera intrinsics
    intrinsic = o3d.camera.PinholeCameraIntrinsic(256, 256, fx, fy, cx, cy)

    # Convert to point cloud
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsic)


    # Create transformation matrix for mirroring around x-axis
    transformation_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Transform point cloud
    pcd.transform(transformation_matrix)
    
    return pcd


if not os.path.exists('depth_images'):
    os.makedirs('depth_images')

if not os.path.exists('point_clouds'):
    os.makedirs('point_clouds')

# Simulation
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.join(currentdir, "../gym")
os.sys.path.insert(0, parentdir)
cid = p.connect(p.SHARED_MEMORY)
if (cid < 0):
  p.connect(p.GUI)
p.resetSimulation()
p.setGravity(0, 0, -10)
useRealTimeSim = 1

p.setRealTimeSimulation(useRealTimeSim)
p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), basePosition=[0, 0, 0])


# Car
car = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "racecar/racecar.urdf"))
inactive_wheels = [5, 7]
wheels = [2, 3]
for wheel in inactive_wheels:
  p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
steering = [4, 6]


# Function to check if a new barrier is too close to existing ones
def too_close(new_barrier, existing_barriers, min_distance=1.25):
    for barrier in existing_barriers:
        dist = math.sqrt((new_barrier[0]-barrier[0])**2 + (new_barrier[1]-barrier[1])**2)
        if dist < min_distance:
            return True
    return False

# List to hold generated barriers
BARRIER = []

while len(BARRIER) < 45:  # Create 25 barriers
    new_barrier = [random.uniform(2, 38), random.uniform(-1, 1)]
    if not too_close(new_barrier, BARRIER):
        BARRIER.append(new_barrier)

# Cylinder shape for the barriers
cylinder_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.15, height=0.5)

# Generate the barriers in the simulation
for coordinate in BARRIER:
    box_body = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=cylinder_shape,
        basePosition=[coordinate[0], coordinate[1], 0.3],
    )

# Move to Point function
def moveTo(targetX, targetY):
  pos, hquat = p.getBasePositionAndOrientation(car)
  h = p.getEulerFromQuaternion(hquat)
  x = pos[0]
  y = pos[1]
  distance = math.sqrt((targetX - x)**2 + (targetY - y)**2)
  theta = math.atan2((targetY - y), (targetX - x))
  i = 0

  while distance > 0:
    pos, hquat = p.getBasePositionAndOrientation(car)
    h = p.getEulerFromQuaternion(hquat)
    x = pos[0]
    y = pos[1]
    distance = math.sqrt((targetX - x)**2 + (targetY - y)**2)
    theta = math.atan2((targetY - y), (targetX - x))
    maxForce = 20
    targetVelocity = 5*distance

    # velocity cap
    if targetVelocity > 20:
       targetVelocity = 20


    steeringAngle = theta - h[2]
    if steeringAngle > (math.pi / 2) or steeringAngle < -(math.pi / 2):
       steeringAngle = h[2] - theta
    else:
       steeringAngle = theta - h[2]


    view_matrix_car = p.computeViewMatrix(
       cameraEyePosition = [pos[0] + 0.5*math.cos(h[2]), pos[1] + 0.5*math.sin(h[2]), pos[2] + 0.1],
       cameraTargetPosition = [pos[0] + 2*math.cos(h[2]), pos[1] + 2*math.sin(h[2]), pos[2] + 0.05],
       cameraUpVector = [0, 0, 1]
    )
    projection_matrix_car = p.computeProjectionMatrixFOV(
      fov=60,  # field of view
      aspect=1.0,  # aspect ratio
      nearVal=0.1,  # near clipping plane
      farVal=100.0  # far clipping plane
    )

    img_arr = p.getCameraImage(256, 256, viewMatrix=view_matrix_car, projectionMatrix=projection_matrix_car, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    depth_buf = np.reshape(img_arr[3], (256, 256))
    depth_img = 1 - np.array(depth_buf)



    # Save depth image
    img_filename = os.path.join('depth_images', f'depth_img_{i}.png')
    cv2.imwrite(img_filename, (depth_img * 255).astype(np.uint8))

    # Convert depth image to point cloud
    # Assume camera intrinsics fx, fy, cx, cy
    fx, fy, cx, cy = 128, 128, 128, 128  # Replace with your camera intrinsics
    pcd = depth_to_pcd(depth_img, fx, fy, cx, cy)

    # Save point cloud
    pcd_filename = os.path.join('point_clouds', f'pcd_{i}.ply')
    o3d.io.write_point_cloud(pcd_filename, pcd)

    i += 1

    
    firsthalf = depth_img[0][:128]
    totalfirsthalf = sum(firsthalf)
    secondhalf = depth_img[0][128:]
    totalsecondhalf = sum(secondhalf)
    middle = depth_img[0][96:160]
    totalmiddle = sum(middle)

    print("First: ", totalfirsthalf)
    print("Second: ", totalsecondhalf)
    print("Middle: ", totalmiddle)
    print("Steering Angle: ", steeringAngle)

    difference = abs(totalfirsthalf - totalsecondhalf)

    if totalmiddle > 4:
      if totalfirsthalf > totalsecondhalf:
          steeringAngle = steeringAngle - 50*difference
          # targetVelocity = targetVelocity * 4
      else:
          steeringAngle = steeringAngle + 50*difference
          # targetVelocity = targetVelocity * 4

    elif difference > 2:  
      if totalfirsthalf > totalsecondhalf:
         steeringAngle = steeringAngle - 6*(1/difference)
      else:
         steeringAngle = steeringAngle + 6*(1/difference)
    

    p.resetDebugVisualizerCamera(
      cameraDistance = 10,
      cameraYaw = -90, 
      cameraPitch = -45,
      cameraTargetPosition = [pos[0] + 5, 0, pos[2]],
      physicsClientId=0
    )


    for wheel in wheels:
        p.setJointMotorControl2(car,
                            wheel,
                            p.VELOCITY_CONTROL,
                            targetVelocity=targetVelocity,
                            force=maxForce)
    for steer in steering:
        p.setJointMotorControl2(car, steer, p.POSITION_CONTROL, targetPosition=steeringAngle)
    if (useRealTimeSim == 0):
        p.stepSimulation()
    time.sleep(0.001)

# Waypoints
points = [(40, 0)]

for i in points:
   moveTo(i[0], i[1])