# Behavioral-Cloning
----
This is a Udacity project aimming at generating neural network code that can learn from your driving behaviors (in simulator) and drive in
the simulator itself. The code takes in images with your steering angle as training data, and save the trained network. When connected to 
the simulator, the network is capable of returning a steering angle according to the curve to drive the car in the simulator.


## Model structure
The model mainly applied a nerual network applied in self-driving car of Nvidia cooperation.The architecture includes 5 convolutional 
layers with ReLu as activation function,a flatten layer and 4 fully connected layers. More detail can be found in the code and writeup.md.

## Model performance
The network trained by the code was able to drive the simulated car in the center of the lane most of the time without leaving the lane
, no matter how the road was curved, and was capable of adjusting itself from over or under turning. The result can be found in the **run.mp4**.
 
 ## File structure
 - Model.py
 
 This is the python file contains the model code and some tunning and testing steps. 
 
 - Model.h5
 
 The trained model was saved in this file.
 
 - writeup.md
 
 Some detailed discussion about the project can be found in this md file.
 
 - UD_README.md
 
 The original readme file of this project.
 
 - drive.py
 
 Code that drives the car in the simulator with the network.
 
 - run.mp4
 
 Video recorded how the network drive the car.
