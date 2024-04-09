import cv2
import numpy as np
from PIL import Image
import random

# Load the Haar cascade file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load an image
image = Image.open("C:/Users/Yugendhar/Downloads/istockphoto-1346125184-1024x1024.jpg")
image_np = np.array(image)
image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

# Perform face detection
gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Count the number of faces detected
num_faces = len(faces)

# List of common names
common_names = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", 
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen"
]

# Generate random names for each face
face_names = random.choices(common_names, k=num_faces)

# Draw rectangles around the detected faces and write names on them
for (x, y, w, h), name in zip(faces, face_names):
    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(image_rgb, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Add text to display the number of faces detected
cv2.putText(image_rgb, "Number of Faces: " + str(num_faces), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Display the result
cv2.imshow('Face Detection', image_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
