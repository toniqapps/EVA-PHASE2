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
