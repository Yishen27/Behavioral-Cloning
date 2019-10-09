**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Figure_1.png "Traning Visualization"
[image2]: ./examples/center.jpg "Center Driving"
[image3]: ./examples/recovery_1.jpg  "Recovery Image"
[image4]: ./examples/recovery_2.jpg "Recovery Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I tried LeNet and the Nvidia Network for my model and chose to use the Nvidia architecture (model.py line 80-94) eventually since it works better. The architecture includes 5 convolutional layers with ReLu as activation function,a flatten layer and 4 fully connected layers. The data was normalized with a lambda layer and images were cropped.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers with the dropout rate 0.5 in order to reduce overfitting (model.py lines 89). 

However, I have one question that I didn't solve here as part of my report.I fully understand the usage of the dropout layer, but I'm not sure where exactly should I put it. Although I know it should be applied to the fully connected layers instead of convolutional layers, I don't know which fully connected layer it should be added to. And I didn't find answer to this with my own search, can you help me with this? Thank you.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 22). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. The model works fairly well driving the simulator.

#### 3. Model parameter tuning

I used an adam optimizer, so the learning rate was not tuned manually (model.py line 97).
The batch size was set to 32 (model.py line 73), and the epoch was tuned to 5 (model.py line 101).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering lap. I recorded two laps of center driving and one recovering lap. Details in the next section. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

At the beginning I tried to use the LeNet architecture since I'm familiar with it. And it works well in the image classification task. And I also tried the Nvidia network, and found out it works better than LeNet (lower training and validation loss). Thus in my model I used the Nvidia network.

In order to evaluate how well the model works, I split my data set into a training (with 80% data) and a validation set (20% data). In order to make the model more efficitent and easy to track the training process, I used a generator function. I found the training and validation loss in my model decreased in the first few epoches but the validation loss increased in the last few epoches. This implied that the model was overfitting. 

To avoid the overfitting, I added a dropout layer to the model. The validation loss decreased with the training loss.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like on the bridge and a sharp turn by the lake. To improve the performance, I recorded a recovering lap accordingly and trained the model again.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. Though the car drove close to the edge at one or two turns, it was able to recover to the center by itself.

#### 2. Final Model Architecture

The final model architecture (model.py line 80-94)) includes 1 lambda layer, 1 corpping layer, 5 convolutional layers with ReLu as activation function,a flatten layer, a dropout layer and 4 fully connected layers. It is basiclly a Nvidia network added a dropout layer. 

From the start, the data was normalized with the lambda layer and cropped to make it only shows the useful road part. Then the convolutional layers contains 3 layers with 5 by 5 kernel, 2 by 2 stride and 24, 36, 48 filters respectively, follows 2 layers with 3 by 3 kernel, 1 by 1 stride and 64 filters. Then it follows a flatten layer, a dropout layer with drop rate of 0.5. Next are fully connected layers, with 100, 50, 10 and 1 nodes respectively. 

#### 3. Creation of the Training Set & Training Process

When collect training data, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

Then I tried to train the model with these data, but found out the car run out of road at some place, so I recorded a recovering lap to enhance the training.
These images show what a recovery looks like.

![alt text][image3]
![alt text][image4]


I devided 20% of the data set into a validation set.

To augment the data, I also set ther flipping in the generator function, this would make the data twice as many and enhance the data for clockwise drive. 

I also randomly shuffled the data set in the generator function. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 after tunning. I found out after 5 epochs the validation loss start to rise again. I used an adam optimizer so that manually training the learning rate wasn't necessary.
