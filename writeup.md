#**Follow Me** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Follow Me Project**

The goals / steps of this project are the following:
* Use the simulator to collect data
* Build, a fully convolution neural network in Keras to perform semantic segmentation of images
* Train and validate the model with a training and validation set
* Test that the model successfully following the target person
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture.png "Model Visualization"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/1155/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model_training.ipynb containing all relevant code to create and train the model
* config_model_weights containing the configuration of the model
* model_weights containing a trained convolution neural network 
* writeup.md or writeup report summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my follower.py file, the the drone will search for target and follow it in case it is identified. 
```sh
python follower.py model_weights
```

####3. Submission code is usable and readable

The model_training.ipynb file contains the code for training and saving the convolution neural network. The file shows the data preparation and pipeline I used for training and validating the model, and it contains comments to explain how the code works.


####1. Network architecture

My model consists of a fully convolution network based on a fully convolutional network as thought in the lessons. A typical convolution neural network (CNN) consist of a series of convolutional layers followed by a fully connected layers and in general a softmax activation function. This architecture is good for classifications like predicting if an image shows a specific object. The limitation of CNN is that it cannot determine where in a picture the specific object is placed as spatial information is not preserved.
Here comes fully convolutional networks into place (FCN). We add several convolutional leayers after the fully connected layer and so we can preserve spatial information. These FCN consist of three main techniques. First the fully connected layers will be replaced by a 1x1 convolutional layer. Second the upsampling of output convolutions will be dne by transpose convolutions and third we can skip connections allowing the network to use information from multiple resolutions scales. I will come to these points later in the report.





Image size matters for CNN as the size of the input is determined by the size of the fully connected layer. In a FCN different sizes does not matter.




from leaning on the network sturcture created by me in the `Behavioral Cloning` Project i have done in Self-Driving Car Nanodegree.





only the input size is different -> 160, 320, 3 (model.py line 103).
This networks consists of 10 layers, including a normalization layer, 5 convolutional layers, and 4 fully connected layers.
The output has size 1 to predict the steering/angle value. (model.py lines 116)

Strided convolutions in the first three convolutional layers with a 2×2 stride are used. After that  a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers are used. (model.py lines 107 - 111)

The model includes ELU layers to introduce nonlinearity (model.py lines 107 - 111), and the data is normalized in the model using a Keras lambda layer (code line 105).

Also a cropping layer is used to reduce complexity of images processed (model.py line 106)


