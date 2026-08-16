import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load input image in grayscale
img = cv2.imread("Exp-4/image.png", cv2.IMREAD_GRAYSCALE)

# Check whether image is loaded
if img is None:
    raise FileNotFoundError("Exp-4/image.png could not be loaded.")

# Generate Gaussian Noise
mean = 0
sigma = 50

noise = np.random.normal(mean, sigma, img.shape)

# Add noise to image
noisy_img = img.astype(np.float32) + noise
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

# Apply different filters

# 1. Mean Filter
mean_filter = cv2.blur(noisy_img, (7, 7))

# 2. Median Filter
median_filter = cv2.medianBlur(noisy_img, 7)

# 3. Gaussian Filter
gaussian_filter = cv2.GaussianBlur(noisy_img, (7, 7), 0)

# 4. Bilateral Filter
bilateral_filter = cv2.bilateralFilter(noisy_img, 9, 75, 75)

# Display results
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(noisy_img, cmap="gray")
plt.title("Gaussian Noisy Image")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(mean_filter, cmap="gray")
plt.title("Mean Filter")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(median_filter, cmap="gray")
plt.title("Median Filter")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(gaussian_filter, cmap="gray")
plt.title("Gaussian Filter")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(bilateral_filter, cmap="gray")
plt.title("Bilateral Filter")
plt.axis("off")

plt.tight_layout()

# Save output inside Exp-4 folder
plt.savefig("Exp-4/output_exp4.png", dpi=300, bbox_inches="tight")
plt.close()

print("Experiment 4 completed successfully.")
print("Gaussian noise added and different filters applied.")
print("Output saved as: Exp-4/output_exp4.png")