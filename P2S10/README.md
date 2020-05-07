# EVA-PHASE2 ASSIGNMENT 10

## Files
```
1. Assignment10.pynb : Network to train auto car
2. Mask1.png : Map on which we train out car
3. autocar_crop.py : Use to get cropped image based on x,y coordinates, angle and size
4. autocar_env.py : Environment which returns next state, reward, done to indicate target acheived or failure and info for debugging purpose
```

***abstract: we use T3D Architecture to train our auto car network, we provide cropped map image has input to the network and using auto car environment along with action predicted by the network we calculate our next state and reward acheived by implementing current action, we run our network for multiple episodes, each episode indicates an experience using which network improves its predication and achieve the define target***

## Cropping Image
We provide cropped image as state to the nextwork which is cropped from the map on which we trying run our auto car model. Cropped image depends open the x,y coordinates and angle of the car using which we out the required cropped image to the network. Firstly we crop larger area making x,y as center and then we rotate the cropped image based on car angle and then recrop the image to the to get a perfectly cropped image which can resized based on the required input size, When we rotate the image we pad image with dummy pixels and passing rotated image with dummy pixels may lead to unwanted learning therefore we crop larger area from the image and after rotation further crop the image based on the input provided this we get a perfectly cropped image with more information without padding padding dummy pixels

## Auto Car Environment
Auto environment is use to calculate next state, reward and status of current action using which our network predicts accurate actions. Network provides action as input to the environment which is basically angle used to calculate the direction we need to move our car. Based on the action the car can move into sand or non sand area (road or no road) or it may hit a dead end or acheive the required target, for each action taken the environment rewards with a positive or negative value to the network for example if the car moves in a sand area then the reward is negative, if the car moves in non sand are the reward is still negative but with a small negative value so that the car doesn't end in a deadlock and larger negative value if it hits a wall, now based on the reward, next state and done status the network plans its next step and predicates better action with experiences gained from repetative exploration

## T3D Network
We use simple mnist network with approximately 9000 parameters, the input to the network is 32x32 1 channel image for both actor and critic model. We run our T3D model first with 10000 random samples and then start taining with actual network. We cap our rewards to maximum -2000 which terminates the current episodes and reset the environment and start training with new random state.


#Current Problems
1. I am yet to combine my code with Kivy for final result
2. My network is runing but not able to acheive the required outcome
3. My rewards are always negative and not sure why the network is not able to predicte accurate action even after 50000 timsteps. should I train for more time steps
4. I am assigning negative penalty of -2 for sand are should I increase it?
5. I am assigning small negative value of -0.2 for non sand area and postive reward if the distance is less previous distance
6. I feel the cropping method is right and even my enironment is calculating coordinates based on the action provided from the network
7. Is the problem with my network


## Final Assesment

### END GAME 
### Auto Driving Network Using T3D Model

<b>Abstract</b>:
  Using T3D model we need to train a model in such a way that it runs only on road and reach defined destination. To achieve this we use `T3D Actor Critic` model. We pass cropped image of size 32x32 from the map as a state to the network along with orientation from car current position to destination and network returns action (car angle), based on the angle we decide to move the car with a fixed velocity in car direction
  
  
<b>Network:</b>
  We pass cropped image of size 32x32 along with orientation to the T3D Actor Critic Model as state. We use MNIST network with 4500 parameters along with fully connected layers to train our model. Using the state dimension of size 3 network trains the model to reduce actor and critic loss. The actor model returns action of size 1 (car angle) which are passed along with state to the critic model which return max qvalue. In the above network we use state_dim of size 2 and action_dim of size 1
  
<b>Environment:</b>
Auto car environment consist of 4 main function 
1. sample() : outputs random action between -max_action and maxaction
2. randomCord() : outputs random coordinates which act as start point for the car (we always pick non sand coordinates)
3. step() : Calculates car new coordinates, orientation and distance  using action from the network and based on the coordinates we provide cropped image along with orientation as input to the network

<b>Image Cropping</b>
We use this class to crop the image from map based on the x,y coordinates and angle. We always cropped larger area taking x,y as center and then rotate the image then again crop the image to remove padded pixel and then resize 200x200 image in 32x32

<b>Experiment</b>
Experiment 1: In first attempt we tried to pass only cropped image as input to the network and network was trained for around 50,000 steps. I have assigned a negative reward of -1 if the car was on sand and -0.5 (living penalty) if the car was on non sand and -10 if hit the wall and 100 if destination reached, max_action of +- 5 was used to keep the angle in certain range. but I was not able acheive the required target with just cropped image

Experiment 15: After several attempts with cropped image I was not able acheive the required target. Problem was it use to randomly hit move on sand or non sand area and was stable after 100000 steps. 

Experiment 16: I decided to pass orientation as input to the network so that network tries to output action which directs car to the target, after several attempts of optimizing the network by reducing the parameters, i was able to make the car move on the non sand but the movement was not stable the car use spin continously while moving on non sand, I tried different reward values, increasing and decreasing max_action but was never able to make my car stable. My network is able to differentiate between sand and non sand and able to move the car on non sand but I am not sure how to stop car spinning while moving. Below are certain experiment from my final network![Screen Shot 2020-05-07 at 9.55.38 PM.png](:storage/ab1d39f5-6966-48f2-baee-c019450b14fe/09e402de.png)

![Image description](https://github.com/toniqapps/EVA-PHASE2/blob/master/P2S10/Screen%20Shot%202020-05-07%20at%209.55.38%20PM.png)

![Image description](https://github.com/toniqapps/EVA-PHASE2/blob/master/P2S10/Screen%20Shot%202020-05-07%20at%209.55.43%20PM.png)

![Image description](https://github.com/toniqapps/EVA-PHASE2/blob/master/P2S10/Screen%20Shot%202020-05-07%20at%209.55.48%20PM.png)

![Image description](https://github.com/toniqapps/EVA-PHASE2/blob/master/P2S10/Screen%20Shot%202020-05-07%20at%209.55.56%20PM.png)

![Image description](https://github.com/toniqapps/EVA-PHASE2/blob/master/P2S10/Screen%20Shot%202020-05-07%20at%209.56.14%20PM.png)

<b>Parameters</b>
Non Sand Reward : -1
Sand Reward : -0.1
Reduce Distance Reward : 0.5
Destination Reward : 100
Wall Carsh Reward : -5
Max_Action : 5
State_Dim : 3 (Cropped Image, orientation, -orientation)
Action_Dim : 1 (Angle)
Network Parameters : 4,653


<b>Conclusion</b>
I was able to move my car on non sand but failed to understand why my car spins while moving, my network have reached it destination few times but most of the time after moving for a while it just hanged at one particular point. My networks required additional parameters to to stabilize the car but not sure what are the additional parameters needed or is it because of max_action or the way I assign rewards. But would love experiment further to find out how to acheive the required target


