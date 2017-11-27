# **Follow Me**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image2]: ./images/encoder_block.png "Encoder Code"
[image3]: ./images/batch_norm.png "Batch Norm Code"
[image4]: ./images/decoder_block.png "Decoder Code"
[image5]: ./images/different_models.png "Model Architecture tries"
[image6]: ./images/parameters.png "Model Parameters"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/1155/view) individually and describe how I addressed each point in my implementation.  

---
#### Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model_training.ipynb containing all relevant code to create and train the model
* config_model_weights containing the configuration of the model
* model_weights containing a trained convolution neural network 
* writeup.md or writeup report summarizing the results

#### Submission includes functional code
Using the Udacity provided simulator and my follower.py file, the the drone will search for target and follow it in case it is identified. 
```sh
python follower.py model_weights
```

#### Submission code is usable and readable

The model_training.ipynb file contains the code for training and saving the convolution neural network. The file shows the data preparation and pipeline I used for training and validating the model, and it contains comments to explain how the code works.


#### Network architecture

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

For the decoding sections i added four layers `Decoded Layers` 1- 3 and `Output Layer`. The strides and filter sizes were according to the encoded sections. (strides=2 and filters 16, 32, 64 and 128). Each decoder layer consinst of several single layers. First i apply bilinear upsampling layer. This technique takes the average weight of four nearby pixels and calculates new weights for upsampled pixels. With this technique you can size up a layer and calculate the new weights (for new pixels) based on the upsampling technique. After that i apply a concatenated layer resulting in skip connections. This technique is used to preserve important segmentation information from previous encoder layers. As we narrow down the scope in each encoded layer to some features we loose information of the whole big picture when we decode back (It is like compression and decompression techniques where it is not guaranteed that all information are preserved). So skip connections are used to retain information from previous layers by connecting the output of one layer to an input of a non adjacent layer. 

After the two separable_conv2d_batchnorm layers are added as this was a tip in the lessons to add one or more layers. This is done so that the model is able to preserve finer spatial information from the previous layers better.

Following image shows the code snippet for applying a decoder section:
![alt text][image4]



I tried out several other filter depths and layer structures. Following images shows some outcommented model architectures:
![alt text][image5]

All of these architectures lead to worse results. Reducing the layers lead to underfitting. The model was not able to detect the target person accordingly and the overall score tended agianst 0!. Trying different filter sizes lead to overfitting or worse performance in overall score (about 5%). Using a brute force technique to get the needed performance of 40% percent i tried out the filter sizes and layer depths variations.


#### Parameters chosen for the the neural network
Following parametrs were chosen:

![alt text][image6]

Here i applied a brute force technique to find the correct parameters as well. First i changed the `steps_per_epoch` and `validation_steps` based on the total amount of test images and validation images divided by `batch_size`. As i used the given standard test and validation set i do not need to adjust this parameter. But in case a add some more data in future these both parameters need to be adjusted. Here a better solution could be applied in future meaning that a caluclation of the total amount of test and validation data will be done before. `Workers`are defined as processes in parallel. I changed it to 4 as i have the information that the AWS Cloud Instance can handle 4 processes in parallel. But this didn't lead to a significatnt calculation boost. The combination of `batch_size`, `learning_rate` and `num_epochs` was very hard to find out. I tried several different combinations. What got very clear to e was that a smaller batch size (like 32) lead to better results. All above that lead to worse results at least in the parameter scenarios i tried. `learning_rate` and `num_epochs` are linked together. A higher learning rate would result in less needed epochs but the risk of a too fast converging model would increase. This happend to me. So i tried from 100 epochs with very small learning rate to 10 epochs with higher learning rate many scenarios. Some were converging against 37% some were very bad. As the adjustment of `learning_rate` and `num_epochs` is dependend on given data and the wanted behaviour of the model a general advice cannot be given.


#### Concepts in network layers
However, when we wish to feed the output of a convolutional layer into a fully connected layer, we flatten it into a 2D tensor. This results in the loss of spatial information, because no information about the location of the pixels is preserved.


The 1x1 convolution is used here (instead of fully connected layer) as the 

Image size matters for CNN as the size of the input is determined by the size of the fully connected layer. In a FCN different sizes does not matter.


#### Image manipulation

#### Limitations to the neural network




 




