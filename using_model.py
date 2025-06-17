from keras.src.saving import load_model
from keras.src.utils import image_dataset_from_directory, img_to_array
from numpy import expand_dims, argmax
import tensorflow as tf

model = load_model("D:/pycharm_projects/ISL_image/all_models/attempt_1_2.keras")

class_names = [i for i in range(1,9 +1)]
class_names.extend([chr(i) for i in range(65,65+26)])


def prediction_proc(obj):# -> str:
    img_array = img_to_array(obj)
    img_array = expand_dims(img_array, 0)
    pred = model.predict(img_array)
    score = tf.nn.softmax(pred[0])
    #print(f"{pred[0]}\t{score}\t{argmax(score)}")

    out = argmax(score)
    try:
        print(out)
        p = class_names[out]
        return p

    except Exception as e:
        print(f"{out}")



test_ds = image_dataset_from_directory(
    directory="D:\\pycharm_projects\\ISL_image\\ISL_data",
    color_mode = "grayscale",
    #validation_split=0.2,
    #subset="testing",
    batch_size=32,
    image_size=(128, 128),
    seed=29092004)


import matplotlib.pyplot as plt

plt.figure(figsize=(17, 17))
plt.title("attempt_1_2.keras\nexpected , predicted\n")
for images, labels in test_ds.take(1):
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"{class_names[labels[i]]} , {prediction_proc(images[i])}")
        plt.axis("off")

plt.show()
