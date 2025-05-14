import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

img_height = 1920
img_width = 1080
num_channels = 3
num_classes = 2
batch_size = 32

model = keras.Sequential([
    keras.Input(shape=(img_height, img_width, num_channels)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)), # reduce the spatial dimensions of the feature maps
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)), # reduce the spatial dimensions of the feature maps
    layers.Flatten(), # flatten the 3D output to 1D
    layers.Dense(num_classes, activation="softmax") 
])

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
#in short the model.compile is used to configure the model for training.
# The optimizer is used to update the model's weights during training.
# the loss function simply measures how well the model is performing.
# The metrics parameter is used to specify the metrics to be evaluated during training and testing.
train_ds = image_dataset_from_directory(
    "./dataset/train",
    labels='inferred',
    label_mode='int',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
# The image_dataset_from_directory function is used to load images from a directory and create a dataset.


val_ds = image_dataset_from_directory(
    "./dataset/val",
    labels='inferred',
    label_mode='int',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
# The image_dataset_from_directory function is used to load images from a directory and create a dataset.

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
# The model.fit function is used to train the model on the training data and validate it on the validation data.
model.save("saved_model/my_model_16x16.keras")

#the relu function:"f(x) = max(0, x)", meaning it outputs the input directly if it's positive else 0
#RECTIFIED LINEAR UNIT ReLU : f(x)=max(0,x) here x is the input to the neuron
# neuron is a basic unit of a neural network
# None linear as it outputs 0 for negative values
#allows non linearity in the model
#allows the model to learn complex patterns in the data
# the dense layer uses the softmax activation function to output a probability distribution over the model outputs
# the softmax function converts the output of the dense layer into a probability distribution
# the classes are the possible outputs of the model
# dense layer connects every neuron in the previous layer to every neuron in the current layer
