from flask import Flask, flash, render_template, request, session, redirect
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import shutil 
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard

%load_ext tensorboard



app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    """
    :return: The home web page.
    """

    ## this will download the data from the internet, it may fail if the server is not up
    ![ -f testDataSmall.npz ] || wget -O testDataSmall.npz "https://www.win.tue.nl/~cdecampos/testDataSmall.npz"
    ![ -f trainDataSmall.npz ] || wget -O trainDataSmall.npz "https://www.win.tue.nl/~cdecampos/trainDataSmall.npz"

    X_train = read_train_data[0]
    y_train = read_train_data[1]
    X_test = read_test_data[0]
    y_test = read_test_data[1]

    
    tensorboard = TensorBoard(log_dir = 'dir_1')

    model().fit(X_train, y_train, batch_size=32, epochs=1, validation_data = (X_test, y_test), callbacks = [tensorboard])
    

    if request.method == 'POST': 
        if request.form["button"] == "Predict":
            data = X_test[0]
            X_test[0] = data
            prediction = model.predict(X_test)
            best = max(prediction[0])
            final = np.where(prediction[0] = best)
            return render_template("test.html", data = final)
    else:
        return render_template("WebsiteEyeDoctor.html")

if __name__ == "__main__":
    app.run(debug=True)

#Some extra function definitions
#Get train data from the server
def read_train_data():
    start_time = time.time()
    print("Start Read Train Data")
    data = np.load("trainDataSmall.npz")
    print("Train data read --- %s seconds ---" % (time.time() - start_time))
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    print("Training - Total examples per class", np.sum(Y_train, axis=0))
    return [X_train, Y_train]

#Get test data from the server
def read_test_data():
    start_time = time.time()
    print("Start Read Test Data")
    data = np.load("testDataSmall.npz")
    print("Test data read --- %s seconds ---" % (time.time() - start_time))
    X_test = data["X_test"]
    Y_test = data["Y_test"]
    print("Testing - Total examples per class", np.sum(Y_test, axis=0))
    return [X_test, Y_test]

#Define the model
def model():
      """Create a Keras Sequential model with layers."""
      model = keras.models.Sequential()
      model.add(Conv2D(32,
                      kernel_size=(3,3),
                      activation='relu',
                      input_shape=X_train.shape[1:]))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      model.add(Dropout(0.25))

      model.add(Conv2D(64,
                      (3, 3),
                      activation='relu'))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      model.add(Dropout(0.15))

      model.add(Flatten())
      model.add(Dense(128, activation='relu'))
      model.add(Dropout(0.3))
      model.add(Dense(5, activation='softmax'))
      model.summary()
      model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adagrad(lr=0.01),
                    metrics=['accuracy'])
      return model
