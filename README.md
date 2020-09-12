# Dog Breed Classification Project. 
This project is a multi-class image classification.  We'll use TensorFlow to build and train a Neural Network that will classify dog breed from images.
Yon can see below some predictions made by the final trained model.




# 🐶  Classify Different Dog Breeds  🐶

To do this, we'll be using data from the [Kaggle dog breed identification competition](https://www.kaggle.com/c/dog-breed-identification/overview). It consists of a collection of 10,000+ labelled images of 120 different dog breeds.

We're going to go through the following workflow:

1. Get data ready (download from Kaggle, store, import).
2. Prepare the data (Turn into tensors & batches, train & valid & test sets)
3. Choose and fit/train a model ([TensorFlow Hub](https://www.tensorflow.org/hub), `tf.keras.applications`, [TensorBoard](https://www.tensorflow.org/tensorboard), [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)).
4. Evalaute the model a model (making predictions, comparing them with the ground truth labels).
5. Improve the model through experimentation 


For preprocessing our data, we're going to use TensorFlow 2.x. The whole premise here is to get our data into Tensors (arrays of numbers which can be run on GPUs) and then allow a machine learning model to find patterns between them. For our machine learning model, we're gonna do some **transfer learning** and we're going to use a pretrained deep learning model from TensorFlow Hub. 

# mobilenet_v2_130_224 Model
In this project, we're using the **`mobilenet_v2_130_224`** model from TensorFlow Hub.
https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html
MobileNetV2 is a significant improvement over MobileNetV1 and pushes the state of the art for mobile visual recognition including classification, object detection and semantic segmentation. MobileNetV2 is released as part of TensorFlow-Slim Image Classification Library, or you can start exploring MobileNetV2 right away in Colaboratory. Alternately, you can download the notebook and explore it locally using Jupyter. MobileNetV2 is also available as modules on TF-Hub, and pretrained checkpoints can be found on github.
<img src="https://github.com/AmineAgrane/dog_breed_classification/blob/master/docs/mobilnetv2.png">

# Performances and results
<img src="https://github.com/AmineAgrane/dog_breed_classification/blob/master/docs/conf_matrix_valid_data.png">
