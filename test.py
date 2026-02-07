import cv2
import numpy as np
import torch

img = cv2.imread("dus.jpeg")
img = cv2.resize(img, (224,224))


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#tells OpenCV Convert from BGR format to Grayscale

gray_norm = gray.astype("float32") / 255.0 #pixel normalization ->ML models cannot work properly with uint8 and this converts uint8 to float32

blur = cv2.GaussianBlur(gray, (5,5), 0)#Apply Gaussian Blur on a grayscale image to reduce noise and smooth the image
edges = cv2.Canny(blur, 100, 200)

# Classical features
edge_density = edges.mean() / 255.0
mean_intensity = gray.mean() / 255.0
std_intensity = gray.std() / 255.0

classical_features = np.array([
    edge_density,
    mean_intensity,
    std_intensity
])

print("Classical features:", classical_features)

# CNN input
cnn_input = gray_norm.reshape(1, 1, 224, 224)
cnn_tensor = torch.tensor(cnn_input, dtype=torch.float32)

print("CNN tensor shape:", cnn_tensor.shape)
