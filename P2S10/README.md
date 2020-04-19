# EVA-PHASE2 ASSIGNMENT 10

## Files
```
1. Assignment10.pynb : Network to train auto car
2. Mask1.png : Map on which we train out car
3. autocar_crop.py : Use to get cropped image based on x,y coordinates, angle and size
4. autocar_env.py : Environment which returns next state, reward, done to indicate target acheived or failure and info for debugging purpose
```

Abstract: We use T3D Architecture to train our auto car network, we provide cropped map image has input to the network and using auto car environment and based on the action predicted by the network we calculate our next state and reward acheived by implementing current action, we run out net work multiple episodes with maximum time steps, each episode indicate an experience using which network improves its predication and achieve the define target
