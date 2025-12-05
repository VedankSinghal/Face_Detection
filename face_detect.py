import cv2
import os
import sys

def detect_faces():
    # 1. Define input and output filenames
    input_image_path = 'input.jpg'
    output_image_path = 'output.jpg'

    # 2. Check if input image exists
    if not os.path.exists(input_image_path):
        print(f"Error: {input_image_path} not found in the repository!")
        sys.exit(1)

    # 3. Load the pre-trained Haar Cascade classifier for face detection
    # We use the one included in cv2 data
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # 4. Read the image
    img = cv2.imread(input_image_path)
    
    # Convert to grayscale (Face detection works better on grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 5. Detect faces
    # scaleFactor=1.1: Reduces image size by 10% each scale
    # minNeighbors=5: How many neighbors each candidate rectangle should have to retain it
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    print(f"Found {len(faces)} faces!")

    # 6. Draw rectangles around the faces
    for (x, y, w, h) in faces:
        # Draw a rectangle: (image, start_point, end_point, color, thickness)
        # Color is BGR: (0, 255, 0) is Green
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 7. Save the result
    cv2.imwrite(output_image_path, img)
    print(f"Successfully saved output to {output_image_path}")

if __name__ == "__main__":
    detect_faces()
