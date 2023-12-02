import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam

import numpy as np

IMG_WIDTH = 178
IMG_HEIGHT = 218

face_haar_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
model = load_model("./models/xception.hdf5", compile=False)
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

gender = ("Man", "Woman")

cam = cv2.VideoCapture(0)
while True :
    image = cam.read()[1]
    faces = face_haar_cascade.detectMultiScale(image, minNeighbors=5)
    for (x,y,w,h) in faces :
            face_image = image[y:y+h, x:x+w]
            cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,255), 1)
            face_image = cv2.resize(face_image, (IMG_HEIGHT, IMG_WIDTH))
            face_image = img_to_array(face_image)
            face_image /= 255
            result = model.predict(np.expand_dims(face_image, axis=0))
            result_index = np.argmax(result)
            g = gender[result_index]
            print(g)
            
            cv2.putText(image, g, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    cv2.imshow("result", image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()