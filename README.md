# 🐶  Dog Breed Classification Project. 🐶   
This project is a multi-class image classification.  I have used TensorFlow [`mobilenet_v2_130_224`](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4) to build and train a Neural Network that will classify dog breed from images passed as input.


Yon can see below some predictions made by the final model. The following predicted labels have been made on some Test images (Images that have not been passed to the Neural Network in its training phase).

<img src="https://github.com/AmineAgrane/dog_breed_classification/blob/master/docs/predicted_labels.png">

<img src="https://github.com/AmineAgrane/dog_breed_classification/blob/master/docs/predicted_labels3.png">

# Classify Different Dog Breeds 

To do this, we'll be using data from the [Kaggle dog breed identification competition](https://www.kaggle.com/c/dog-breed-identification/overview). It consists of a collection of 10,000+ labelled images of 120 different dog breeds.

We're going to go through the following workflow:

1. Get data ready (download from Kaggle, store, import).
2. Prepare the data (Turn into tensors & batches, train & valid & test sets)
3. Choose and fit/train a model ([TensorFlow Hub](https://www.tensorflow.org/hub), `tf.keras.applications`, [TensorBoard](https://www.tensorflow.org/tensorboard), [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)).
4. Evalaute the model a model (making predictions, comparing them with the ground truth labels).
5. Improve the model through experimentation 


For preprocessing our data, we're going to use TensorFlow 2.x. The whole premise here is to get our data into Tensors (arrays of numbers which can be run on GPUs) and then allow a machine learning model to find patterns between them. For our machine learning model, we're gonna do some **transfer learning** and we're going to use a pretrained deep learning model from TensorFlow Hub. 

# Getting Data ready
## Preprocessing images (Turning images to Tensors)

To preprocess our images into Tensors we're going to :
1. Uses TensorFlow to read the file and save it to a variable, `image`.
2. Turn our `image` (a jpeg file) into Tensors.
3. Normalize image (from 0-255 to 0-1)
4. Resize the `image` to be of shape (224, 224).
5. Return the modified `image`.

A good place to read about this type of function is the [TensorFlow documentation on loading images](https://www.tensorflow.org/tutorials/load_data/images). 

## Creating data batches

Dealing with 10,000+ images may take up more memory than your GPU has. Trying to compute on them all would result in an error. So it's more efficient to create smaller batches of your data and compute on one batch at a time.

# Creating and training the Neural Network 
## mobilenet_v2_130_224 Model

In this project, we're using the **`mobilenet_v2_130_224`** model from TensorFlow Hub.
https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html
MobileNetV2 is a significant improvement over MobileNetV1 and pushes the state of the art for mobile visual recognition including classification, object detection and semantic segmentation. MobileNetV2 is released as part of TensorFlow-Slim Image Classification Library, or you can start exploring MobileNetV2 right away in Colaboratory. Alternately, you can download the notebook and explore it locally using Jupyter. MobileNetV2 is also available as modules on TF-Hub, and pretrained checkpoints can be found on github.
<img src="https://github.com/AmineAgrane/dog_breed_classification/blob/master/docs/mobilnetv2.png">

## Setting up the model layers

The first layer we use is the model from TensorFlow Hub (`hub.KerasLayer(MODEL_URL)`. This **input layer** takes in our images and finds patterns in them based on the patterns [`mobilenet_v2_130_224`](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4) has found.

The next layer (`tf.keras.layers.Dense()`) is the **output layer** of our model. It brings all of the information discovered in the input layer together and outputs it in the shape we're after, 120 (the number of unique labels we have). The `activation="softmax"` parameter tells the output layer, we'd like to assign a probability value to each of the 120 labels [somewhere between 0 & 1](https://en.wikipedia.org/wiki/Softmax_function). The higher the value, the more the model believes the input image should have that label. 


# Performances and results
## Confusion Matrix
The following confusion matrix is the result of using our model to classify the validation dataset images. The model have been trained on 9000 images, and the validation dataset have 1000 images.
<img src="https://github.com/AmineAgrane/dog_breed_classification/blob/master/docs/conf_matrix_valid_data.png">

## Predicted Label/Breed and probability distribution
Another way to analyse the inference of our model is to print out the predicted class of an image and it's probability distribution. For each image, we show it's original label (left), it's predicted label (right), and the probabilty assossiated with the predicted label (how much confident is our Neural Network about the predicted class).

<img src="https://github.com/AmineAgrane/dog_breed_classification/blob/master/docs/predicted_labels2.png">


# Improve the model
How to approuve model accuracy :
1. [Trying another model from TensorFlow Hub](https://tfhub.dev/) - A different model could perform better on our dataset. 
2. [Data augmentation](https://bair.berkeley.edu/blog/2019/06/07/data_aug/) - Take the training images and manipulate (crop, resize) or distort them (flip, rotate) to create even more training data for the model to learn from. 
3. [Fine-tuning](https://www.tensorflow.org/hub/tf2_saved_model#fine-tuning)
