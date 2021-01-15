from flask import Flask, flash, render_template, request, session, redirect, url_for
from werkzeug.utils import secure_filename
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
import PIL
from PIL import Image
from numpy import asarray



app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    """
    :return: The home web page.
    """

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

    testdata = np.load("testDataSmall.npz")
    traindata = np.load("trainDataSmall.npz")

    X_train = traindata["X_train"]
    y_train = traindata["Y_train"]
    X_test = testdata["X_test"]
    y_test = testdata["Y_test"]

    tensorboard = TensorBoard(log_dir = 'dir_1')

   
    

    if request.method == 'POST': 
        if "butTrain" in request.form:
             model().fit(X_train, y_train, batch_size=32, epochs=1, validation_data = (X_test, y_test), callbacks = [tensorboard])
             return render_template("WebsiteEyeDoctor.html",data = None)
        elif "button" in request.form:
            data = X_test[0]
            file = request.files['file']

            
            if file:
                filename = secure_filename(file.filename)
                file.save(filename)
                image = Image.open(filename)
                data = asarray(image)


            

            X_test[0] = data
            prediction = model().predict(X_test)
            best = max(prediction[0])
            final = np.where(best)
            return render_template("WebsiteEyeDoctor.html", data = final[0]+1)
    else:
        return render_template("WebsiteEyeDoctor.html", data = None)

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


