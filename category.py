from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

classes = ["Good", "Average", "Bad"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("condition_model.pt"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

img = Image.open("dus.jpeg")
img = transform(img).unsqueeze(0)

outputs = model(img)
probs = torch.softmax(outputs, dim=1)

print("Prediction:", classes[probs.argmax()])
print("Confidence:", probs.max().item())
