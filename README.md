Certainly! To make the code more organized when pasted on GitHub, you can use headings and subheadings. Here's a modified version of the code with appropriate headings:

```python
# Image Classification - Happy and Sad

## Importing Libraries
```python
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model
```

## GPU Configuration
```python
# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
print("====================GPU's List============================")
print("GPUS : ", gpus)
print("================================================\n")
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')
```

## Data Preprocessing
```python
# Remove dodgy images
# ...

# Load Data
data = tf.keras.utils.image_dataset_from_directory('data') # Its a generator function
# ...
```

## Data Visualization
```python
# Visualization
# ...
```

## Data Preprocessing
```python
# Preprocessing
# ...
```

## Splitting Data
```python
# Split Data
# ...
```

## Model Architecture
```python
# Build Deep Learning Model
model = Sequential()
# ...
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()
```

## Model Training
```python
# Train
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
```

## Plot Performance
```python
# Plot Performance

# Loss
fig = plt.figure()
# ...

# Accuracy
fig = plt.figure()
# ...
plt.legend(loc="upper left")
plt.show()
```

## Model Evaluation
```python
# Model Evaluation
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
# ...
print(pre.result(), re.result(), acc.result())
```

## Model Testing
```python
# Test
img = cv2.imread('happy_pic.jpg')
# ...
if yhat > 0.5: 
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')
```

## Save and Load Model
```python
# Save the Model
model.save(os.path.join('models','imageclassifier.h5'))
new_model = load_model('imageclassifier.h5')
new_model.predict(np.expand_dims(resize/255, 0))
```

Feel free to adjust the headings based on your preferences or add more details as needed. This structure should help in making the code more readable on GitHub.
