import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import numpy as np
import tensorflow as tf
import PIL
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers


print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_accuracy')>0.98):
      print("\nReached 98% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

# Directory with our training horse pictures
train_horse_dir = os.path.join('./data/training/horses')
# Directory with our training human pictures
train_human_dir = os.path.join('./data/training/humans')
# Directory with our training horse pictures
validation_horse_dir = os.path.join('./data/validation/horses')
# Directory with our training human pictures
validation_human_dir = os.path.join('./data/validation/humans')

train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)

print("Definimos directorio con las imagenes para evaluar..")
test_dir = os.path.join('./data/test')
test_file_names = os.listdir(test_dir)


print("TAMAÑO DEL SET DE IMAGENES")
print('total training horse images:', len(os.listdir(train_horse_dir)))
print('total training human images:', len(os.listdir(train_human_dir)))
print("-------------------------")

IMG_SIZE = 300

resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMG_SIZE, IMG_SIZE),
  layers.Rescaling(1./255)
])

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2),
])

model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])



model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        './data/training/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        './data/validation/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

model.summary()



history = model.fit(
      train_generator,
      validation_data=validation_generator,
      epochs=15,
      steps_per_epoch=8,
      validation_steps=8,
      verbose=1,
      callbacks=[callbacks])

print("\n-------------------------------")
print("   GRAFICAMOS EL APRENDIZAJE   ")
print("-------------------------------")

history_dict = history.history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.close()
plt.figure()
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
plt.close()



print("Definimos directorio con las imagenes para evaluar..")

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        './data/test/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=10,
        class_mode='binary')

image, label = test_generator.next()

for i in range(len(image)):
    my_image = np.expand_dims(image[i], 0)
    result = model.predict(my_image)

    if result < 0.5:
            print("\nPREDICCION: Es Caballo:    valor:" + str(result))
    else:
            print("\nPREDICCION: Es Humano :    valor:" + str(result))

    if label[i] == 0:
            print("REAL      : Es Caballo: ")
    else:
            print("REAL      : Es Humano : ")

    plt.imshow(image[i])
    plt.show()

model.save('saved-model/clasify_hoh')