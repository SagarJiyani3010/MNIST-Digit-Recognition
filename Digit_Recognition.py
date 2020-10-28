import tensorflow as tf
import matplotlib.pyplot as plt

class myCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.99):
            print("\nYou reached to 99% acuuracy, so training has been stopped...!")
            self.model.stop_training = True

data = tf.keras.datasets.mnist

(X_train, y_train),(X_test, y_test) = data.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

callbacks = myCallbacks()

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation = tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation = tf.nn.softmax)])

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 10, callbacks = [callbacks])

print("\n Accuracy = ", end="")
print(model.evaluate(X_test, y_test))