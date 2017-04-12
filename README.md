# Behavioral Cloning

### Submitted by: Juan silva

---

## Objectives

The goals / steps of this project are the following:  
* Use the simulator to collect data of good driving behavior    
* Build, a convolution neural network in Keras that predicts steering angles from images  
* Train and validate the model with a training and validation set  
* Test that the model successfully drives around track one without leaving the road  


## About the files in this project

**model.py:** Python script that loads the dataset, creates a convolutional network model, trains it and saves it to disk  

**mymodels.py:** Python script that contains helper functions to create my keras models, illustrating the pipeline for each model (imported in model.py)

**drive.py:** Used to load the trained model and send steering angle predictions to the Unity car simulator in autonomous mode

**model.h5:** The trained convolutional neural network (saved from model.py)

**writeup_report.pdf:** Written report of the project

**video.mp4:** Video showing one full loop of autonomous driving

## Running the code

The generated model can be used to run the simulator in autonomous mode. For this, use the generated networ in model.h5 and the drive.py script:

```sh
python drive.py model.h5
```

Once the program start, open the simulator and start autonomous mode.


## Network Architecture

The final implementation was achieved through iterations trying different approaches. The final implementation found in model.py uses a network architectured from Nvidia's work "End to End Learning for Self-Driving Cars" published here: 
[https://arxiv.org/pdf/1604.07316v1.pdf](https://arxiv.org/pdf/1604.07316v1.pdf)  
![alt text][image1]

This architecure was chosen because the simmilarities between the DAVE-2 system and the Simulator used in this project. Specifically, the use of three cameras for and end to end training and control of steering angle.

In the network there is a total of 9 layers. Starting from the bottom, the first layer is a normalization layer. Having normalization as part of the network model permits acceleration using the GPU processing.

Then we have 5 convolutional layers. They were design to do feature extraction. The first three of those are striding convolutions using a 2x2 stride and a 5x5 kernel. The last two are non strided convolutions with a kernel of 3x3. After each convolution a RELU activation is used to introduce non nonlinearity.

After the flattening of those convolutional layers, there are 3 fully connected layers that output a final control value for the stering angle. 

## Model and Strategy

### Training Data

The network was trained with data from behavioural clonning using the provided driving simulator. This is a still from the simulator screen:

![alt text][image2]

Recording driving behaviour in training mode the simulator generates images from three different camera angles (left, center and right) along with a measurement of steering angle.

![alt text][image3]  
![alt text][image4]  
![alt text][image5]

For training we used data from all three cameras, using a correction factor for left and right cameras which teached the network how to dive back to the center when drifting off the road. The correction factor was set to 2 degrees of offset steering opposite to the side of the camera. 

Data was further augmented by flipping the images horizontally and changing the sign of the corresponding steering angle (i.e. multiplied by -1)



### Training strategy

For training, data was devided between training and validation using an 80/20 split. model.py uses a python generator to process the data in batches of size 32, which is in turn fed to the keras model.

For normalization, a lambda layer of the keras framework was used. This step involved adding a layer that would devide each pixel value by its max and moved to the center:

```sh
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
```

As their were processed, images were also cropped to focus on the middle part that includes the road area. This way the featured at the top of the image that include sky, trees etc was exluded from the training process. See the following two images for before / after cropping operation.

![alt text][image4]
![alt text][image6]  

As part of the generator logic, sample data is shuffled trhough each batch being processed.

Adam optimizer was used to minimize the mean squared error of the steering angle output and the angle clonned from driver behaviour. The network was trained through 5 epochs.


## Autonomous Testing

The output network from our solution approach was tested by using the autonomous driving mode of the simmulator. 

Initial tests included data from two loops recorded. This caused the car to more or less stay in the road, but it kept going too close to the left border of the road which eventually made it go off road. To augment the training date, 2 more loops were recorded but going in the opposite direction on the track.

This kept the car in the road, but it still failed to take the sharp curve at the end of the loop. The car would just drive straight not not follow much of the curve.

After this more data was collected on that are of the road, another run included two passes to that curve.

With this new knowledge of how to manouver that curve, the model was able to make the car go around the track multiple times without going off the road.

The file video.mp4 contains the final run with a full loop of autonomous driving aroudn track 1.






[//]: # (Image References)

[image1]: ./report_images/network.png "Model Visualization"
[image2]: ./report_images/simulator.png "Simulator"
[image3]: ./report_images/left.jpg "Left Camera"
[image4]: ./report_images/center.jpg "Left Camera"
[image5]: ./report_images/right.jpg "Left Camera"
[image6]: ./report_images/center_cropped.jpg "Center Camera Cropped"
[image7]: ./examples/placeholder_small.png "Flipped Image"

