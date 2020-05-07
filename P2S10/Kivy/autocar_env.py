# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import autocar_crop

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image
from kivy.graphics.texture import Texture
import threading
import torch

# Importing the Dqn object from our AI in ai.py
import TD3

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '714')
Config.set('graphics', 'height', '330')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
last_reward = 0
scores = []
img2 = Image.open('./images/MASK1.png').convert('L')
img = Image.open('./images/mask.png').convert('L')
action2rotation = [0,5,-5]
# textureMask = CoreImage(source="./kivytest/simplemask1.png")

## We create the policy network (the Actor model)
policy = TD3.TD3(1, 1, 5)
# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur, largeur))
    sand = np.asarray(img)/255
    goal_x = 100
    goal_y = 560
    first_update = False
    global swap
    swap = 0
    policy.load('TD3_AutoCarEnv_0','./pytorch_models/')


# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.angle = self.angle + rotation
        


x_array = []
y_array = []
# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.pos = Vector(800, 400)
        self.car.velocity = Vector(0, 0)

    #Random cordinates assigned to x and y initially or on reset
    def randomCord(self):
        self.car.x = 800
        self.car.y = 400

    def step(self, action):
        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        
        self.done = 0
        longueur = self.width
        largeur = self.height
        if first_update:
            self.car.x = 10
            self.car.y = largeur - 250
            init()

        
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        
        cropped_image = autocar_crop.croppedImage(img2, self.car.y, self.car.x, self.car.angle, 32)
        action = policy.select_action(np.asarray(cropped_image)/255, np.asarray([0, 0]))
        
        self.car.move(int(action))
        
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        
        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y))
            
            last_reward = -1
        else: # otherwise
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            last_reward = -0.1
            print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y))
            if distance < last_distance:
                last_reward = 0.5
            # else:
            #     last_reward = last_reward +(-0.2)

        #if car hits the wall assign high penalty and done = 1
        if (self.car.x < 5 or self.car.x > self.width - 5 or self.car.y < 5 or self.car.y > self.height - 5):
            self.done = 1
            print('crashed')
            last_reward = -5

        if distance < 25:
            last_reward = 100
            self.done = 1
            if swap == 1:
                goal_x = 800
                goal_y = 400
                swap = 0
            else:
                goal_x = 100
                goal_y = 560
                swap = 1
        last_distance = distance

    #reset environment with default and new random x,y coordinates and cropped image with x,y has center
    def reset(self):
        self.randomCord()
        cropped_image = autocar_crop.croppedImage(img, self.car.y, self.car.x, 0, 32)
        return np.asarray(cropped_image)/255, np.asarray([0, 0])
    
    # return random angle rangign min to max
    def sample(self):
        return np.random.randint(-10, 10, size=1)[0]

class CarApp(App):
    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.step, 1.0/60.0)
        return parent

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
