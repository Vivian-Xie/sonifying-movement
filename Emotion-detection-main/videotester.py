import os
import cv2
import numpy as np
import keras.utils as image
import warnings
warnings.filterwarnings("ignore")
from keras.utils import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np
import json

from socket import *
print("client is activated")
host_name="10.209.92.109"
#host_name="192.168.99.132"
#host_name="127.0.0.1"
#host_name="220.196.194.163"
#host_name="192.168.142.214"
port_num=1200
clientSocket=socket(AF_INET,SOCK_STREAM)
clientSocket.connect((host_name,port_num))

"""
message=input("enter:")
clientSocket.send(message.encode())
while True:
    upperMessage=clientSocket.recv(1024).decode()
    if upperMessage:
        print("message from the server: "+upperMessage)

clientSocket.close()
"""









# load model
model = load_model("D:/sonify/Emotion-detection-main//best_model.h5")


face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


list=[]
filename='emotion-storage.json'


cap = cv2.VideoCapture(0)
while True:
    ret, test_img = cap.read()  # captures frame and returns boolean v alue and captured image
    if not ret:
        list+=[" "]
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
 
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255
        #print("this is a test")
        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        if predicted_emotion:
            list+=[predicted_emotion]
            if len(list)==30:
                list=[]
                clientSocket.send(predicted_emotion.encode())
                print("message sent",predicted_emotion)
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        with open(filename, 'w') as f1:
            json.dump(list, f1)
        print(list)
        break
        

cap.release()
cv2.destroyAllWindows