# CarND-Traffic-Sign-Classifier-
# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/distribution.PNG "Distribution"
[image2]: ./images/data_image.PNG "Data Image"
[image3]: ./images/architecture.PNG "Architecture"
[image4]: ./images/test_image.PNG "Test Original Images"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  
The project code can be found in [Traffic_Sign_Classifier.ipynb.](Traffic_Sign_Classifier.ipynb) 

---
### Data Set Summary & Exploration

The dataset used for this project is [German Traffic Sign Dataset](https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) which cabn be downloaded from [here.](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)
![alt text][image2]

- Number of training examples = 34799
- Number of valid examples= 4410
- Number of testing examples = 12630
- Image data shape = (32, 32, 3)
- Number of classes = 43

Here's a image for distributions of classes.
![alt text][image1]


### Design and Test a Model Architecture

#### 1.  Image data preprocessing 

1. The image data was normalized so that the data has mean zero and equal variance. For image data, (pixel - 128)/ 128 is a quick way to approximately normalize the data and can be used in this project.
2. Grayscale conversion - image color is not a distinguishing feature for traffic signs. 

#### 2.Model Architecture

My final model consisted of the following layers:
![alt text][image3]

#### 3. Description of my training model

My training model was used Stochastic Gradient Descent optimized by AdamOptimizer at a learning rate of 0.001. Each batch was a 128 training samples and the loss converged for the validation set at 20 epochs.
The probability to drop out the specific weight during training was KEEP_PROB  = 0.7.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.950
* validation set accuracy of 0.950 
* test set accuracy of 0.968

My first model was the LeNet architecture with excellent handwriting recognition performance.
All conditions were the same as for LeNet, but the training accuracy was 0.85.
Changed the filter depth of the first layer from 6 to 12, and increased it from 16 to 25 for the second layer. The accuracy has been increased to about 0.92.

Then, as a regulatory method to prevent overfitting, I added a dropout layer after the second layer. I kept the parameter at 0.5. As a result, the accuracy was increased to 0.95.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 

The second image, "Bicycles crossing", can be difficult to classify because the surroundings are dark. The third image can be difficult to classify as it contains text with a certain speed limit in the middle. When the brightness of a photo is too bright or too dark, and at 32x32 resolution, it is difficult to tell.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Bicycles crossing		| Turn left ahead								|
| Speed limit (30km/h)  | Keep right									|
| Wild animals crossing	| Wild animals crossing			 				|
| Yield      			| Yield             							|


Despite achieving an accuracy of 0.95 on the test set, the accuracy of the new traffic sign was 0.6. I think it can be solved by using more image preprocessing techniques or augmentation features etc in the training set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a stop sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop sign   									| 
| .64     				| Turn left ahead								|
| .88					| Keep right									|
| 1.0	      			| Wild animals crossing			 				|
| 1.0				    | Yield             							|
