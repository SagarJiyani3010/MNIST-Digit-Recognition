import tensorflow as tf

class myCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.99):
            print("\nYou reached to 99% acuuracy, so training has been stopped...!")
            self.model.stop_training = True

data = tf.keras.datasets.mnist

(X_train, y_train),(X_test, y_test) = data.load_data()

X_train = X_train.reshape(len(X_train), 28, 28, 1) / 255.0
X_test = X_test.reshape(len(X_test), 28, 28, 1) / 255.0

callbacks = myCallbacks()

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 10, callbacks = [callbacks])

print("\n Accuracy = ", end="")
print(model.evaluate(X_test, y_test))