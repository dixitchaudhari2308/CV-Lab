import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load grayscale image
image_path = Path(__file__).parent / "image.png"
image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found.")
    exit()

# Resize for faster processing
image = cv2.resize(image, (600, 400))

# ---------------- Salt and Pepper Noise ----------------
def add_salt_pepper_noise(image, probability=0.35):
    noisy = image.copy()
    random_matrix = np.random.rand(*image.shape)

    noisy[random_matrix < probability / 2] = 0
    noisy[random_matrix > 1 - probability / 2] = 255

    return noisy

noisy_image = add_salt_pepper_noise(image)

# ---------------- Filters ----------------
mean_result = cv2.blur(noisy_image, (7, 7))
median_result = cv2.medianBlur(noisy_image, 7)

# ---------------- Kernel Size Comparison ----------------
k3 = cv2.medianBlur(noisy_image, 3)
k7 = cv2.medianBlur(noisy_image, 7)
k11 = cv2.medianBlur(noisy_image, 11)

# ---------------- Stride Comparison ----------------
s1 = cv2.medianBlur(noisy_image, 5)[::1, ::1]
s2 = cv2.medianBlur(noisy_image, 5)[::2, ::2]
s4 = cv2.medianBlur(noisy_image, 5)[::4, ::4]

print("Stride 1 Shape:", s1.shape)
print("Stride 2 Shape:", s2.shape)
print("Stride 4 Shape:", s4.shape)

# ---------------- Main Output ----------------
plt.figure(figsize=(16, 5))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(noisy_image, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(mean_result, cmap="gray")
plt.title("Mean Filter 7x7")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(median_result, cmap="gray")
plt.title("Median Filter 7x7")
plt.axis("off")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "output_main.png")
plt.show()

# ---------------- Kernel Output ----------------
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(k3, cmap="gray")
plt.title("Kernel 3x3")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(k7, cmap="gray")
plt.title("Kernel 7x7")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(k11, cmap="gray")
plt.title("Kernel 11x11")
plt.axis("off")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "output_kernel.png")
plt.show()

# ---------------- Stride Output ----------------
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(s1, cmap="gray")
plt.title(f"Stride 1\nShape: {s1.shape}")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(s2, cmap="gray")
plt.title(f"Stride 2\nShape: {s2.shape}")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(s4, cmap="gray")
plt.title(f"Stride 4\nShape: {s4.shape}")
plt.axis("off")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "output_stride.png")
plt.show()

print("Experiment 3 completed successfully.")