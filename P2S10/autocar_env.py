import gym
from gym import error, spaces, utils
from gym.utils import seeding

import autocar_crop
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np 

from kivy.vector import Vector

#Read map on which we are training our car to drive
img = Image.open('MASK1.png').convert('L')
#width and height of image
width, height = img.size
#this is to load initial data
first_update = True

'''
Default values to be loaded 
goal_x, goal_y target which the car as to reach from source
first_update: if true load initial data
sand : image mask used to validate sand and non sand values
swap: swap the destination once target reach
'''
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    goal_x = 1420
    goal_y = 622
    sand = np.zeros((height, width))
    sand = np.asarray(img)/255
    first_update = False
    global swap
    swap = 0

# Initializing the last distance
last_distance = 0
last_reward = 0

'''
Autocar Environment where input to network is cropped image and 
output is angel using which we decide car direction
state_dim :  1 input (Cropped image) 
action_dim: 1 ouput (angle output from the network)
max_action (max angle range)
low and high (min and max noise)
x and y (car position)
randomCord : initialize x and y coordinates with random coordinates initially
'''
class AutoCarEnv(object):  
    def __init__(self):
        self.state_dim = 1
        self.action_dim = 1
        self._max_episode_steps = 1000
        self.velocity = (2, 0)
        self.done = 0
        self.max_action = 30
        self.low = -5
        self.high = 5
        self.angle = 0
        self.x = 0
        self.y = 0
        self.randomCord()

    #Random cordinates assigned to x and y initially or on reset
    def randomCord(self):
        self.x = np.random.randint(0, high=width-1, size=1, dtype='l')[0]
        self.y = np.random.randint(0, high=height-1, size=1, dtype='l')[0]

    #new car postion and angle based on output from then network
    def move(self, rotation):
        self.x, self.y = Vector(*self.velocity) + [self.x, self.y]
        self.angle = self.angle + rotation
    
    '''
    step to be taken to find the next state and assign reward based on the action taken
    input is action from the network used to calculate car direction
    if on sand assign negative reward of 2
    if not on sand assign living penalty reward of negative 0.2
    if not on sand and closer to destination assign postive reward of 0.5
    if hit the wall assign negative reward -10
    output is 
        next_state : next observation (cropped image) 
        reward : reward based on the action taken
        done : done to indicate if current action led to achieve required 
               target or hit a wall or 0 if exploring
        info : debug info
    '''
    def step(self, action):
        global goal_x
        global goal_y
        global last_reward
        global last_distance
        global swap

        if first_update:
            init()

        self.done = 0

        xx = goal_x - self.x
        yy = goal_y - self.y
        
        self.move(action)
        
        distance = np.sqrt((self.x - goal_x)**2 + (self.y - goal_y)**2)

        if sand[int(self.y),int(self.x)] > 0:
            self.velocity = Vector(0.5, 0).rotate(self.angle)
            #print(1, goal_x, goal_y, distance, int(self.x),int(self.y))
            last_reward = -2
        else:
            self.velocity = Vector(2, 0).rotate(self.angle)
            last_reward = -0.2
            #print(0, goal_x, goal_y, distance, int(self.x),int(self.y))
            if distance < last_distance:
                last_reward = 0.5

        #if car hits the wall assign high penalty and done = 1
        if (self.x < 5 or self.x > width - 5 or self.y < 5 or self.y > height - 5):
            self.done = 1
            last_reward = -10

        # swap the destination once car reaches current destination
        if distance < 25:
            self.done = 1
            if swap == 1:
                goal_x = 1420
                goal_y = 622
                swap = 0
            else:
                goal_x = 9
                goal_y = 85
                swap = 1
        last_distance = distance
        
        # cropped image with center x,y and output size 32 and rotate angle to align with car direction
        cropped_image = autocar_crop.croppedImage(img, self.x, self.y, self.angle, 32)
        #return cropped image, reward acheived for the action taken 
        return np.asarray(cropped_image), last_reward, self.done, {}

    # return random angle rangign min to max
    def sample(self):
        return np.random.randint(-10, 10, size=1)[0]

    #reset environment with default and new random x,y coordinates and cropped image with x,y has center
    def reset(self):
        self.randomCord()
        cropped_image = autocar_crop.croppedImage(img, self.x, self.y, 0, 32)
        return np.asarray(cropped_image)

    #random seed value
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]