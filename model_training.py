import tensorflow as tf
from keras.src.utils import image_dataset_from_directory
from keras.src.losses import SparseCategoricalCrossentropy

from keras.src.models.sequential import Sequential
from keras.src.layers import Rescaling, RandomFlip, RandomZoom, RandomRotation, Conv2D, Flatten, MaxPooling2D, Dense

train_ds = image_dataset_from_directory(
    directory="D:\\pycharm_projects\\ISL_image\\ISL_data",
    color_mode = "grayscale",
    validation_split=0.2,
    subset="training",
    batch_size=32,
    image_size=(128, 128),
    seed=29092004)

val_ds = image_dataset_from_directory(
    directory="D:\\pycharm_projects\\ISL_image\\ISL_data",
    color_mode="grayscale",
    validation_split=0.2,
    subset="validation",
    batch_size=32,
    image_size=(128, 128),
    seed=29092004)

class_names = train_ds.class_names
print(len(class_names))
print(class_names)


AUTOTUNE = tf.data.AUTOTUNE

trains_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(25):
#         ax = plt.subplot(5, 5, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
# plt.show()

data_augmentation = Sequential([
    RandomFlip("vertical",input_shape=(128,128,1)),
    RandomRotation(0.2),
    RandomZoom(0.1),
])


model = Sequential([
    Rescaling(1./255, input_shape=(128, 128, 1)),
    data_augmentation,
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_names))
])

model.compile(optimizer="adam",
              loss= SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.summary()

epoch = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoch
)

# model.save("D:\\pycharm_projects\\ISL_image\\all_models\\second.keras")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Define plot labels and styles
plot_labels = ['Accuracy', 'Loss']
plot_styles = ['-', '--']

# Plot training and testing accuracy/loss
for i, metric in enumerate(['accuracy', 'loss']):
    train_metric = history.history[metric]
    test_metric = history.history['val_' + metric]
    axs[i].plot(train_metric, label='Training ' + metric.capitalize(), linestyle=plot_styles[0])
    axs[i].plot(test_metric, label='Testing ' + metric.capitalize(), linestyle=plot_styles[1])
    axs[i].set_xlabel('Epochs')
    axs[i].set_ylabel(plot_labels[i])
    axs[i].legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
