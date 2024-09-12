# Import necessary libraries
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load models
mymodel = load_model('./models/AGNew.h5')
Emotion_model = load_model("./models/mymodel.h5")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define emotion labels
emotion_labels = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Define a function to process the uploaded image
import base64

# Define a function to process the uploaded image
def process_image(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    predictions = []
    for (x, y, w, h) in faces:
        # Extract face region
        face_img = img[y:y+h, x:x+w]
        # Resize face image for age and gender model
        face_resized = cv2.resize(face_img, (200, 200))
        normalized_face = face_resized / 255.0
        normalized_face = np.expand_dims(normalized_face, axis=0)
        # Resize face image for emotion model
        face_resized1 = cv2.resize(face_img, (48, 48))
        normalized_face1 = face_resized1 / 255.0
        normalized_face1 = np.expand_dims(normalized_face1, axis=0)
        # Predict age and gender
        pred_age = int(mymodel.predict(normalized_face)[0][0][0])
        pred_gender = mymodel.predict(normalized_face)[1][0][0]
        # Predict emotion
        pred_emotion = Emotion_model.predict(normalized_face1)[0]
        # Append predictions
        predictions.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "age": pred_age,
            "gender": pred_gender,
            "emotion": emotion_labels[np.argmax(pred_emotion)]
        })
    return predictions



# Define route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        predictions = process_image(img)
        
        # Draw bounding boxes and predictions on the image
        for i, prediction in enumerate(predictions, start=1):
            x, y, w, h = prediction['x'], prediction['y'], prediction['w'], prediction['h']
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
            age = int(prediction['age'])
            gender = "Female" if prediction['gender'] > 0.5 else "Male"
            label = f"{i}"
            cv2.putText(img, label, (x, y + h + 35), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 10)

        # Encode the image to JPEG format
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        # Render the HTML template with the processed image and predictions
        return render_template('result.html', img_base64=img_base64, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
