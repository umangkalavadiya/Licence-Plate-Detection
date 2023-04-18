# Licence-Plate-Detection
This code is an implementation of license plate detection using TensorFlow and Keras. It uses a pre-trained NASNetLarge model for feature extraction and then adds some fully connected layers for classification.

The code first reads XML files containing information about the bounding boxes of the license plates in each image. Then it preprocesses the images by resizing them and normalizing their pixel values. The bounding boxes are also normalized by dividing the x and y coordinates by the width and height of the image, respectively.

The preprocessed images and normalized bounding boxes are used to train the model using the mean squared logarithmic error loss function and the Adam optimizer. The model is then evaluated on a validation set and saved to a file.

Finally, a function is defined for detecting license plates in a given image using the saved model. It loads the image and preprocesses it in the same way as the training data. The preprocessed image is then passed to the model to predict the bounding box coordinates of the license plate. The predicted coordinates are denormalized and used to draw a rectangle around the license plate in the original image.
