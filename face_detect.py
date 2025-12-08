import cv2
import os
import sys

def detect_faces(input_path='frame_1.png', output_path='output.jpg'):
    # 1. Check if input image exists
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found!")
        return 0  # Return 0 faces found if file missing

    # 2. Load the classifier
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # 3. Read and process image
    img = cv2.imread(input_path)
    if img is None:
        print("Error: Failed to load image.")
        return 0
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    num_faces = len(faces)
    print(f"Found {num_faces} faces!")

    # 5. Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 6. Save result
    cv2.imwrite(output_path, img)
    
    # Return the number of faces so our Test file can check it!
    return num_faces

if __name__ == "__main__":
    detect_faces()
