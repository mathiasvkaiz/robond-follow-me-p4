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

[image1]: ./images/model.jpg "Model Visualization"
[image2]: ./images/encoder_block.jpg "Encoder Code"
[image3]: ./images/batch_norm.jpg "Batch Norm Code"
[image4]: ./images/decoder_block.jpg "Decoder Code"

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

Following image shows my final model architecture:

![alt text][image1]

The `Inputs` layer is the resized image from the chopter. After this layer i have added three separabale convolution laysers `Encoded Layer` 1 - 3. A separable layer is taken out of the lessons and is a technique to recude overall parameters that needs to be computed. In difference to regular convolution this separable convolutions do not apply all wanted kernels on each channel but use one kernel per channel and afterwards the the desired kernel number is applied on a 1x1 convolution. This results in far less parameters which leads to better computational times and makes the model more robust to overfitting due to reduced parameters. Based on the experiences i have made in the `Behavioral Cloning` Project of Self-Driving Car Nanodegree i have added more layers here. So for each separable convolution i have added a regular convolution with different kernel size (5x5 instead 3x3) for each encoded layer. Following images shows the code part where this is applied:

![alt text][image2]

This lead to a better final score of nearly 8%. I tried `elu` instead of `relu` activation but this lead to a worse overall performance of min. 2% and more. I have used strides=2 for all encoded layers and a filter size from 16 up to 128 as this resulted in best results. Changing those parameters lead to worse performance up to nearly 20% (depending on layer depth and stride value).

After applying the three encoded layers i used a batch normalization layer including an 1x1 convolution to `Normed` Layer.
This `Normed` layers applies an 1x1 convolution together with batch normalization. Following images shows the code snippet:

![alt text][image3]

Here also `relu` activation is used as the `elu` activation resulted in 4% less performance.

Batch normalization is used for normalizing layers within the network. Each layer's inputs is normalized by using the mean and variance. This helps to prevent overfitting and the network trains faster. 

For the decoding sections i added four layers `Decoded Layers` 1- 3 and `Output Layer`. The strides and filter sizes were according to the encoded sections. (strides=2 and filters 16, 32, 64 and 128)

Following image shows the code snippet for applying a decoder section:
![alt text][image4]






 However, when we wish to feed the output of a convolutional layer into a fully connected layer, we flatten it into a 2D tensor. This results in the loss of spatial information, because no information about the location of the pixels is preserved.


The 1x1 convolution is used here (instead of fully connected layer) as the 

Image size matters for CNN as the size of the input is determined by the size of the fully connected layer. In a FCN different sizes does not matter.





