#{'cars': 0, 'cats': 1, 'dogs': 2, 'person': 3, 'planes': 4}
from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
#from keras.optimizers import Adam
# from keras.optimizers import adam_v2
# opt = adam_v2.Adam()

from keras.models import model_from_json
from tkinter import simpledialog

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tkinter import messagebox
import cv2
from imutils import paths
import imutils



main = tkinter.Tk()
main.title("Image Classification Using CNN") #designing main screen
main.geometry("600x500")

global filename
global loaded_model

def upload(): #function to upload tweeter profile
    global filename
    filename = filedialog.askopenfilename(initialdir="testimages")
    messagebox.showinfo("File Information", "image file loaded")
    

def generateModel():
    global loaded_model
    if os.path.exists('model.json'):
        with open('model.json', "r") as json_file:
           loaded_model_json = json_file.read()
           loaded_model = model_from_json(loaded_model_json)

        loaded_model.load_weights("model_weights.h5")
        loaded_model._make_predict_function()   
        print(loaded_model.summary)
        messagebox.showinfo("Model Generated", "CNN Model Generated on Train & Test Data. See black console for details")
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(32, 3, 3, input_shape = (48, 48, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 128, activation = 'relu'))
        classifier.add(Dense(output_dim = 5, activation = 'softmax'))
        classifier.compile(optimizer = 'opt', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        train_datagen = ImageDataGenerator()
        test_datagen = ImageDataGenerator()
        training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size = (48, 48),
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
						 shuffle=True)
        test_set = test_datagen.flow_from_directory('data/validation',
                                            target_size = (48, 48),
                                            batch_size = 32,
                                            class_mode = 'categorical',
					    shuffle=False)
        classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 1,
                         validation_data = test_set,
                         nb_val_samples = 2000)
        classifier.save_weights('model_weights.h5')
        model_json = classifier.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        print(training_set.class_indices)
        print(classifier.summary)
        messagebox.showinfo("Model Generated", "CNN Model Generated on Train & Test Data. See black console for details")

def classify():
    imagetest = image.load_img(filename, target_size = (48,48))
    imagetest = image.img_to_array(imagetest)
    imagetest = np.expand_dims(imagetest, axis = 0)
    preds = loaded_model.predict(imagetest)
    print(str(preds)+" "+str(np.argmax(preds)))
    predict = np.argmax(preds)
    msg = ""
    if predict == 0:
        msg = "Image Contains Car"
    if predict == 1:
        msg = "Image Contains Cat"
    if predict == 2:
        msg = "Image Contains Dog"
    if predict == 3:
        msg = "Image Contains Person"
    if predict == 4:
        msg = "Image Contains Plane"
    if predict == 5:
        msg = "Image not trained"
    
    imagedisplay = cv2.imread(filename)
    orig = imagedisplay.copy()
    output = imutils.resize(orig, width=400)
    cv2.putText(output, msg, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    cv2.imshow("Predicted Image Result", output)
    cv2.waitKey(0)


def exit():
    global main
    main.destroy()
    
font = ('times', 16, 'bold')
title = Label(main, text='Image Classification Using Deep Learning CNN Algorithm', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 14, 'bold')
model = Button(main, text="Generate CNN Train & Test Model", command=generateModel)
model.place(x=200,y=100)
model.config(font=font1)  

uploadimage = Button(main, text="Upload Test Image", command=upload)
uploadimage.place(x=200,y=150)
uploadimage.config(font=font1) 

classifyimage = Button(main, text="Classify Picture In Image", command=classify)
classifyimage.place(x=200,y=200)
classifyimage.config(font=font1) 

exitapp = Button(main, text="Exit", command=exit)
exitapp.place(x=200,y=250)
exitapp.config(font=font1) 

main.config(bg='light coral')
main.mainloop()
