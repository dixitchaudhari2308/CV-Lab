import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load Image
img = cv2.imread("Exp-1/image.png")

if img is None:
    print("Image not found!")
    exit()

print("Image Shape:", img.shape)
print("Image Data Type:", img.dtype)


# 1. Original Image
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 2. RGB Image
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(rgb)
plt.title("RGB Image")
plt.axis("off")
plt.show()


# 3. Image Information
height, width, channels = img.shape

print("\nImage Information")
print("Height:", height)
print("Width:", width)
print("Channels:", channels)


# 4. Read Pixel
print("\nPixel Value at (150,150):")
print(img[150, 150])


# 5. Modify Pixel
modified = img.copy()
modified[150, 150] = [255, 0, 0]

cv2.imshow("Modified Pixel", modified)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 6. Region of Interest
roi = img[80:280, 80:280]

cv2.imshow("Region of Interest", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 7. Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Grayscale Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 8. Resize
resized = cv2.resize(img, (500, 350))

cv2.imshow("Resized Image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 9. Flip
horizontal = cv2.flip(img, 1)
vertical = cv2.flip(img, 0)

cv2.imshow("Horizontal Flip", horizontal)
cv2.imshow("Vertical Flip", vertical)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 10. Rotate
rotated = cv2.rotate(img, cv2.ROTATE_180)

cv2.imshow("Rotated Image", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 11. Draw Rectangle
rectangle = img.copy()

cv2.rectangle(
    rectangle,
    (80, 80),
    (350, 280),
    (255, 0, 0),
    2
)

cv2.imshow("Rectangle", rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 12. Add Text
text = img.copy()

cv2.putText(
    text,
    "OpenCV Demo",
    (40, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 255, 255),
    2
)

cv2.imshow("Image with Text", text)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nExperiment 1 completed successfully.")