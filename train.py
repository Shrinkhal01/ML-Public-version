import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

img_height = 720
img_width = 1280
num_channels = 3
num_classes = 2
batch_size = 32

model = keras.Sequential([
    keras.Input(shape=(img_height, img_width, num_channels)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(num_classes, activation="softmax")
])

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

train_ds = image_dataset_from_directory(
    "./train",
    labels='inferred',
    label_mode='int',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    "./val",
    labels='inferred',
    label_mode='int',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

model.save("saved_model/my_model.keras")