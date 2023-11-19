Image Classification - Happy and Sad
This code is an implementation of an image classification task using TensorFlow and Keras. The goal is to train a deep learning model to classify images into two classes: "Happy" and "Sad." The dataset is assumed to be organized into two directories, one for each class.

Prerequisites
TensorFlow (imported as tf)
OpenCV (cv2)
NumPy (np)
Matplotlib (plt)
Setup and GPU Configuration
The code begins by configuring GPU memory growth to avoid out-of-memory errors. It lists available GPUs and sets memory growth for each.

Data Preprocessing
There is a section of code (currently commented out) for removing potentially problematic images from the dataset based on their file extensions. The dataset is loaded using tf.keras.utils.image_dataset_from_directory, and basic visualizations of the data are shown.

Data Preprocessing and Splitting
The dataset is preprocessed by normalizing pixel values to a range between 0 and 1. It is then split into training, validation, and test sets.

Model Architecture
The deep learning model is a convolutional neural network (CNN) implemented using the Sequential API of Keras. It consists of convolutional layers with max pooling and dropout for regularization, followed by dense layers. The final layer uses the sigmoid activation function for binary classification.

Model Training
The model is compiled using the Adam optimizer and binary crossentropy loss. It is trained on the training set for 20 epochs, with validation on a separate validation set. Training progress is logged for later visualization.

Performance Visualization
After training, the code generates plots for both loss and accuracy over epochs.

Model Evaluation
Precision, recall, and binary accuracy metrics are calculated on the test set after training.

Model Testing
The code includes a section for testing the trained model on a new image (happy_pic.jpg). The image is loaded, resized, and the model predicts whether it belongs to the "Happy" or "Sad" class.

Model Saving and Loading
The trained model is saved to a file (imageclassifier.h5). The code then loads the saved model and performs a prediction on the same image.

Note: The code includes commented-out sections for visualization and image removal, which can be uncommented and executed based on specific needs. Additionally, make sure to provide a suitable dataset directory and adjust parameters as needed for your specific use case.
