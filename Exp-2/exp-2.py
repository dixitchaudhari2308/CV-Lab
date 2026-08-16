import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load Image
image = cv2.imread(r"Exp-2/image.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()

print("Image loaded successfully.")
print("Image Shape :", image.shape)
print("Data Type :", image.dtype)

# Display Original Image
plt.figure(figsize=(7, 5))
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()

# Salt and Pepper Noise
noise_probability = 0.35

random_matrix = np.random.uniform(0, 1, image.shape)

salt_mask = random_matrix < (noise_probability / 2)
pepper_mask = random_matrix > (1 - noise_probability / 2)

noisy_image = image.copy()

noisy_image = np.where(salt_mask, 255, noisy_image)
noisy_image = np.where(pepper_mask, 0, noisy_image)

noisy_image = noisy_image.astype(np.uint8)

# Display Noisy Image
plt.figure(figsize=(7, 5))
plt.imshow(noisy_image, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")
plt.show()

# Save Result
cv2.imwrite(r"Exp-2/output_exp2.png", noisy_image)

print()
print("Noise Probability :", noise_probability)
print("Noisy image generated successfully.")
print("Output saved : Exp-2/output_exp2.png")
print()
print("Experiment 2 completed successfully.")