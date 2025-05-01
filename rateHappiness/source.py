import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
import glob
import cv2 as cv



from mega import Mega
from mega import Mega

mega = Mega()
email = "raspberrypi2357@gmail.com"
password = "hello@2357"
print("Connecting to Mega")
m = mega.login(email, password)

try:
    files=m.get_files_in_node('CoZCBRiB')
    for file in files:
        f=(file,files[file])
        # print(f)
        m.download(f, 'C:/Users/sooda/Desktop/Folders/Cprogfiles/.vscode/PYTHON/rateHappiness/input')   
except PermissionError:
    pass
except:
    print("error")

print("Data recieved from Mega")
# Path to the dataset folders
smiling_faces_path = 'C:/Users/sooda/Desktop/Folders/Cprogfiles/.vscode/PYTHON/rateHappiness/train_data/smile'
non_smiling_faces_path = 'C:/Users/sooda/Desktop/Folders/Cprogfiles/.vscode/PYTHON/rateHappiness/train_data/non_smile'
files = glob.glob ("C:/Users/sooda/Desktop/Folders/Cprogfiles/.vscode/PYTHON/rateHappiness/input/*.jpg")

# Load smiling faces
smiling_faces = []
for file in os.listdir(smiling_faces_path):
    img = cv2.imread(os.path.join(smiling_faces_path, file))
    img = cv2.resize(img, (64, 64))
    smiling_faces.append(img)

# Load non-smiling faces
non_smiling_faces = []
for file in os.listdir(non_smiling_faces_path):
    img = cv2.imread(os.path.join(non_smiling_faces_path, file))
    img = cv2.resize(img, (64, 64))
    non_smiling_faces.append(img)

# Convert the lists to NumPy arrays
smiling_faces = np.array(smiling_faces)
non_smiling_faces = np.array(non_smiling_faces)

# Create labels (1 for smiling faces, 0 for non-smiling faces)
smiling_labels = np.ones(smiling_faces.shape[0])
non_smiling_labels = np.zeros(non_smiling_faces.shape[0])

# Concatenate the data and labels
data = np.concatenate((smiling_faces, non_smiling_faces), axis=0)
labels = np.concatenate((smiling_labels, non_smiling_labels), axis=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='BinaryCrossentropy', metrics=['accuracy'])
#model.compile(optimizer='adam', loss='BinaryFocalCrossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

#input files
images = np.empty(len(files))

outputs=[]
for i in files:
        test_image= cv.imread(i)
        img = test_image
        face_image=test_image
        
        # convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # read haar cascade for face detection
        face_cascade = cv2.CascadeClassifier('C:/Users/sooda/Desktop/Folders/Cprogfiles/.vscode/PYTHON/rateHappiness/haarcascades/haarcascade_frontalface_default.xml')

        # Detects faces in the input image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # loop over all the faces detected
        for (x,y,w,h) in faces: 
            face_image = img[y:y+h, x:x+w]
            
        test_image=face_image
        
        #show for reference
        
        test_image = cv2.resize(test_image, (64, 64))

        test_image = np.array(test_image)

        # Reshape the image 
        test_image = test_image.reshape(1, 64, 64, 3)

        # Make a prediction
        prediction = model.predict(test_image)
        
        #print(prediction)
        # Convert the prediction to a 1-10 points
        
        if prediction[0][0] <=0.0001:
            lev=1
        elif prediction[0][0]<=0.1:
            lev=2
        elif prediction[0][0]<=0.2:
            lev=3
        elif prediction[0][0]<=0.3:
            lev=4
        elif prediction[0][0]<=0.4:
            lev=5
        elif prediction[0][0]<=0.6:
            lev=6
        elif prediction[0][0]<=0.7:
            lev=7
        elif prediction[0][0]<=0.8:
            lev=8
        elif prediction[0][0]<=0.9999:
            lev=9
        elif prediction[0][0]<=1:
            lev=10

        # Print the predicted label
        print("Prediction:", lev ,prediction[0][0])
        outputs.append(lev)
print(outputs)



