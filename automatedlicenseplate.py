import cv2
import easyocr
import matplotlib.pyplot as plt

# Load the image
image_path = 'car.jpg'  # Replace with your image path
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load pre-trained Haar cascade for license plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# Initialize OCR reader
reader = easyocr.Reader(['en'])

for (x, y, w, h) in plates:
    plate_img = image[y:y+h, x:x+w]
    text = reader.readtext(plate_img)

    # Draw bounding box and add recognized text
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if text:
        detected_text = text[0][1]
        cv2.putText(image, detected_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

# Convert BGR to RGB for matplotlib display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display image
plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.axis('off')
plt.title("License Plate Detection")
plt.show()