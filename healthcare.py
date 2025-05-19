import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_abnormalities(image_path):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"❌ File not found or invalid format: {image_path}")
        return

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Thresholding to detect bright regions (simulate abnormalities)
    _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

    # Find contours of bright regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert grayscale to BGR for color bounding boxes
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Ignore very small regions
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Show result
    plt.figure(figsize=(10, 6))
    plt.title("Detected Abnormalities")
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Change this to your image file name
image_path = "hand_xray.jpg"
detect_abnormalities(image_path)