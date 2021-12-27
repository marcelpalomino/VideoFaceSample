import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt

train_dir = 'E:/faces/train'
test_dir = 'E:/faces/test'
val_dir = 'E:/faces/valid'

EPOCHS = 12
BATCH_SIZE = 128
steps_per_epoch = 10000//BATCH_SIZE
validation_steps = 2000//BATCH_SIZE

train_generator = ImageDataGenerator(rescale=1./255)
valid_generator = ImageDataGenerator(rescale=1./255)

train_data = train_generator.flow_from_directory(batch_size=4, directory=train_dir,
                                                 target_size=(32, 32), class_mode='categorical')
valid_data = valid_generator.flow_from_directory(batch_size=4, directory=val_dir,
                                                 target_size=(32, 32), class_mode='categorical')

print(train_data.class_indices.keys())

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=120, activation='relu'))
model.add(tf.keras.layers.Dense(units=84, activation='relu'))
model.add(tf.keras.layers.Dense(units=5, activation='softmax'))
model.summary()

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
history = model.fit(train_data, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
                    validation_data=valid_data, validation_steps=validation_steps, shuffle=True)
model.save('E:/models/faces.h5')

pd.DataFrame(history.history).plot()
plt.grid()
plt.gca().set_xlim(0, 11)
plt.gca().set_xlabel('epochs')
plt.show()
