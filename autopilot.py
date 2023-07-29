import pygame
import math
from queue import PriorityQueue
import pybullet as p
import pybullet_data
import math
import numpy as np
import time
import random
import json
import matplotlib.pyplot as plt

# PYBULLET

# Tuning Parameters
translation = [5, 9]
stretching = [1.5, 1]

gap_threshold = 40  # min gap a car fits through
car_buffer = 20  # close obstacles
steering_constant = 0.005  # nerf correction
frames = 2  # takes snapshots every {frames} seconds
capture = True  # boolean to save images

zone_width = 30  # obstacle meters
zone_depth = 30  # obstacle meters
num_cubes = 30  # Number of cubes to load

dimensions = 25 # resolution
scale = 0.3 # pygame to world coordinates

# Physics
physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.resetSimulation()
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(1)  # either this

# Camera
p.resetDebugVisualizerCamera(
    cameraDistance=0.1,
    cameraYaw=- 90,
    cameraPitch=10,
    cameraTargetPosition=[
        0, 0, 0.2],
    physicsClientId=0
)
# Path
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Assets
p.loadSDF("stadium.sdf")
car = p.loadURDF("racecar/racecar.urdf")

# Define the race position
race_position = [0, 0, 0]  # [x, y, z]

def too_close(new_barrier, existing_barriers, min_distance=0.5):
    for barrier in existing_barriers:
        dist = math.sqrt((new_barrier[0]-barrier[0])**2 + (new_barrier[1]-barrier[1])**2)
        if dist < min_distance:
            return True
    return False

# BARRIER = [[3, 0]]
BARRIER = [[2, -0.1], [3.5, 0.3], [3.7, 0], [4, -1], [5, 1.2]]
# BARRIER = [[3, 0]]

# while len(BARRIER) < 10:  # Create 25 barriers
#     new_barrier = [random.uniform(2, 5), random.uniform(-2, 2)]
#     if not too_close(new_barrier, BARRIER):
#         BARRIER.append(new_barrier)

# Cylinder shape for 7the barriers
cylinder_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.1, height=0.5)

# Generate the barriers in the simulation
for coordinate in BARRIER:
    box_body = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=cylinder_shape,
        basePosition=[coordinate[0], coordinate[1], 0.3],
    )


# Init
start_time = time.time()
elapsed_time = 0

# Wheels
inactive_wheels = [5, 7]
wheels = [2, 3]

# Driving
while (elapsed_time < 0.4):
    print(elapsed_time)
    elapsed_time = time.time() - start_time
    for wheel in wheels:
        p.setJointMotorControl2(car,
                                wheel,
                                p.VELOCITY_CONTROL,
                                targetVelocity=10,
                                force=1)
    pos, hquat = p.getBasePositionAndOrientation(car)
    h = p.getEulerFromQuaternion(hquat)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.1,
        cameraYaw=math.degrees(h[2]) - 90,
        cameraPitch=10,
        cameraTargetPosition=[
            pos[0] + math.cos(h[2]), pos[1] + math.sin(h[2]), pos[2] + 0.2],
        physicsClientId=0
    )

# DEPTH CAMERA
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
cam_info = p.getDebugVisualizerCamera()
view_matrix = cam_info[2]
proj_matrix = cam_info[3]
width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
    256, 256, viewMatrix=view_matrix, projectionMatrix=proj_matrix
)
depth = np.array(depth_img).reshape((256, 256))

# POINT CLOUD
projection = np.array(proj_matrix).reshape(4, 4)
fx = projection[0][0]
fy = -projection[1][1]
cx = projection[0][2]
cy = projection[1][2]

# Generate point cloud
v, u = np.indices(depth.shape)
z = depth.copy()
valid_mask = (z > 0)

x = (u - cx) * z / fx
y = (v - cy) * z / fy + 150
point_cloud = np.column_stack(
    (x[valid_mask], y[valid_mask], 10000 - z[valid_mask] * 10000))

# Convert the angle to radians
theta = math.radians(12)

# Define the rotation matrix around the x-axis
R_x = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta), np.cos(theta)]
])

rotated_points = np.dot(point_cloud, R_x)


# Filter background and ground
obstacles = rotated_points[(rotated_points[:, 2] > 0)
                           & (rotated_points[:, 1] > 0.5)]

# 2D map
map = obstacles.copy()
map[:, 1], map[:, 2] = obstacles[:, 2], obstacles[:, 1]
map[:, 2] = 0
map = map[:, :2]

# Transform map
scaled_map = map * ((dimensions - 1) / 238) # scale it down
local_map = np.array([[((dimensions - 1) / 2) -
                       (x - ((dimensions - 1) / 2)), y] for x, y in scaled_map]).astype(np.int64) # flip the y axis along its middle

local_map[:, 0] += translation[0]
local_map[:, 1] += translation[1]

local_map[:, 0] = (local_map[:, 0] - ((dimensions - 1) / 2)) * stretching[0] + ((dimensions - 1) / 2)
local_map[:, 1] = (local_map[:, 1] - ((dimensions - 1) / 2)) * stretching[1] + ((dimensions - 1) / 2)

# Check bounds
x_condition = (local_map[:, 0] >= 0) & (local_map[:, 0] <= 24)
y_condition = (local_map[:, 1] >= 0) & (local_map[:, 1] <= 24)

# Combine the conditions for x and y using logical AND (&) to get the final condition
final_condition = x_condition & y_condition

# Apply boolean indexing to get the filtered array
local_map = local_map[final_condition]


# Save local_map
def save_map(array, filename):
    plt.clf()
    plt.scatter(array[:, 0], array[:, 1], s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, dimensions - 1)
    plt.ylim(0, dimensions - 1)
    plt.title('Obstacles')
    plt.savefig(filename)
    plt.close()


# Debug
# np.set_printoptions(threshold=np.inf)
# data = {
#     'local_map': local_map.tolist()
# }
# with open('local_map.json', 'w') as f:
#     json.dump(data, f)

# reshaped_depth_img = np.reshape(depth, (256, 256))
                                            
# depth_data = {
#     'depth': reshaped_depth_img.tolist(),
#     'proj_matrix': proj_matrix
# }
# with open('depth_output.json', 'w') as f:
#     json.dump(depth_data, f)

plt.imsave(f'autopilot_img/depth.png', depth)
save_map(local_map, f'autopilot_img/local_map.png')

# ASTAR
WIDTH = 500
ROWS = 25
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

PATH = []
local_map[:, 1] *= -1
local_map[:, 1] += 24

class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_inflation(self):
        return self.color == GREY

    def is_barrierinflation(self):
        return self.color == BLACK or self.color == GREY

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_inflation(self):
        self.color = GREY

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(
            win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        # DOWN
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrierinflation():
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrierinflation():  # UP
            self.neighbors.append(grid[self.row - 1][self.col])

        # RIGHT
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrierinflation():
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrierinflation():  # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

        # DOWN LEFT
        if (self.row < self.total_rows - 1) and (not grid[self.row + 1][self.col].is_barrierinflation()) and (self.col > 0) and (not grid[self.row][self.col - 1].is_barrierinflation()):
            self.neighbors.append(grid[self.row + 1][self.col - 1])

        # DOWN RIGHT
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrierinflation() and self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrierinflation():
            self.neighbors.append(grid[self.row + 1][self.col + 1])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrierinflation() and self.col > 0 and not grid[self.row][self.col - 1].is_barrierinflation():  # UP LEFT
            self.neighbors.append(grid[self.row - 1][self.col - 1])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrierinflation() and self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrierinflation():  # UP RIGHT
            self.neighbors.append(grid[self.row - 1][self.col + 1])

    def __lt__(self, other):
        return False


def inflate_obstacles(grid, inflate_range=1):
    inflation = []
    for row in grid:
        for spot in row:
            if spot.is_barrier():
                inflation.append(spot)

    for spot in inflation:
        obstacle_row, obstacle_col = spot.get_pos()
        for row in range(-inflate_range, inflate_range + 1):
            for col in range(-inflate_range, inflate_range + 1):
                if row == 0 and col == 0:  # Skip the obstacle itself
                    continue
                inflated_row, inflated_col = obstacle_row + row, obstacle_col + col
                if (0 <= inflated_row < len(grid) and
                    0 <= inflated_col < len(grid[0]) and
                        not grid[inflated_row][inflated_col].is_barrier()):  # Make sure spot is in the grid and is not already an obstacle.
                    # Mark the spot as an obstacle
                    grid[inflated_row][inflated_col].make_inflation()


def h(p1, p2):  # manhattan vs eudclidean distance: heuristic function
    x1, y1 = p1
    x2, y2 = p2
    # return abs(x1 - x2) + abs(y1 - y2)
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


# draw path IMPORTANT BECAUSE ROBOT NEED TO FOLLOW THIS
def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        PATH.append([scale * (current.y / 20), scale * ((current.x / 20) - 12)])
        draw()


def reset_path(grid, start, end):
    for row in grid:
        for spot in row:
            if spot != start and spot != end and not spot.is_barrier() and not spot.is_inflation():
                spot.reset()


def algorithm(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))  # add to priority queue
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0  # distance from start
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())  # distance to end

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()  # way to exit the loop

        current = open_set.get()[2]  # starting at the start node
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            if abs(neighbor.row - current.row) == 1 and abs(neighbor.col - current.col) == 1:
                temp_g_score = g_score[current] + math.sqrt(2)
            else:
                temp_g_score = g_score[current] + 1
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + \
                    h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()
        draw()

        if current != start:
            current.make_closed()

    return False


def make_grid(rows, width):  # makes grid
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)

    return grid  # 2 dimensional list


def draw_grid(win, rows, width):  # draws gridlines
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)  # draws each box color

    draw_grid(win, rows, width)  # draws gridlines
    pygame.display.update()  # update display


def main(win, width):
    grid = make_grid(ROWS, width)
    start = None
    end = None

    draw(win, grid, ROWS, width)
    start = grid[12][0]
    start.make_start()
    end = grid[12][24]
    end.make_end()

    for coordinate in local_map:
        grid[coordinate[0]][coordinate[1]].make_barrier()

    reset_path(grid, start, end)
    inflate_obstacles(grid)  # Inflate obstacles before running the algorithm
    for row in grid:
        for spot in row:
            spot.update_neighbors(grid)
    algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)
    # pygame.quit()





def moveTo(targetX, targetY):
    pos, hquat = p.getBasePositionAndOrientation(car)
    h = p.getEulerFromQuaternion(hquat)
    x = pos[0]
    y = pos[1]
    distance = math.sqrt((targetX - x)**2 + (targetY - y)**2)
    theta = math.atan2((targetY - y), (targetX - x))
    while distance > 1:
        pos, hquat = p.getBasePositionAndOrientation(car)
        h = p.getEulerFromQuaternion(hquat)
        x = pos[0]
        y = pos[1]
        distance = math.sqrt((targetX - x)**2 + (targetY - y)**2)
        theta = math.atan2((targetY - y), (targetX - x))
        maxForce = 20
        targetVelocity = 10*distance

        # velocity cap
        if targetVelocity > 20:
            targetVelocity = 20

        steeringAngle = theta - h[2]
        if steeringAngle > (math.pi / 2) or steeringAngle < -(math.pi / 2):
            steeringAngle = h[2] - theta
        else:
            steeringAngle = theta - h[2]
        

        for wheel in wheels:
            p.setJointMotorControl2(car,
                                    wheel,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=targetVelocity,
                                    force=maxForce)

        for steer in steering:
            p.setJointMotorControl2(
                car, steer, p.POSITION_CONTROL, targetPosition=steeringAngle)

        p.resetDebugVisualizerCamera(
            cameraDistance=0.1,
            cameraYaw=math.degrees(h[2]) - 90,
            cameraPitch=10,
            cameraTargetPosition=[
                pos[0] + math.cos(h[2]), pos[1] + math.sin(h[2]), pos[2] + 0.2],
            physicsClientId=0
        )

        p.stepSimulation()
        time.sleep(0.01)


main(WIN, WIDTH)

# AUTOPILOT
for wheel in inactive_wheels:
    p.setJointMotorControl2(
        car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
steering = [4, 6]

PATH.reverse()

# target_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.1, height=4)
# target_pos = PATH[-1]
# p.createMultiBody(
#         baseMass=1.0,
#         baseCollisionShapeIndex=target_shape,
#         basePosition=[target_pos[0], target_pos[1], 2],
#     )

for coordinate in PATH:
    moveTo(coordinate[0], coordinate[1])
time.sleep(3)