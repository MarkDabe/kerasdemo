import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

# load the mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalize the datasets
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#define the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )
#train the model
model.fit(x_train, y_train, epochs=3)

#test the model
validation_loss, validation_accuracy = model.evaluate(x_test, y_test)

print("validation_loss = {}".format(validation_loss))
print("validation_accuracy = {}".format(validation_accuracy))

predictions = model.predict([x_test])



for counter in range(0, predictions.size):
    print(np.argmax(predictions[counter]))
    plt.imshow(x_test[counter], cmap=plt.cm.binary)
    current_plot = plt.show()
    # use keyboard interrupt to exit