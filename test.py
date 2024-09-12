import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Load models
mymodel = load_model('./models/AgeGenderModel.h5')
Emotion_model = load_model("./models/mymodel.h5")

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load test image
img = cv2.imread('test.jpg')

# Convert the image to grayscale for face detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    # Extract face region
    face_img = img[y:y+h, x:x+w]
    
    # Resize face image for age and gender model
    face_resized = cv2.resize(face_img, (150, 150))
    normalized_face = face_resized / 255.0
    normalized_face = np.expand_dims(normalized_face, axis=0)
    
    # Resize face image for emotion model
    face_resized1 = cv2.resize(face_img, (48, 48))
    normalized_face1 = face_resized1 / 255.0
    normalized_face1 = np.expand_dims(normalized_face1, axis=0)
    
    # Predict age and gender
    pred_age = mymodel.predict(normalized_face)[0][0][0]
    pred_gender = mymodel.predict(normalized_face)[1][0][0]
    
    # Predict emotion
    pred_emotion = Emotion_model.predict(normalized_face1)[0]
    
    # Print predictions
    print("Predicted age:", pred_age)
    print("Predicted gender:", pred_gender)
    print("Predicted emotion probabilities:", pred_emotion * 100)
    print("Predicted emotion:", np.argmax(pred_emotion))
    
    # Draw rectangle around the face
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the output image with detected faces
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
