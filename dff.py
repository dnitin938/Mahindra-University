import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tkinter import Tk, filedialog

# ----------- CONFIG -----------
MODEL_PATH = "chromium_cnn_augmented.pth"
IMAGE_SIZE = (128, 128)
NUM_CLASSES = 9
CLASS_NAMES = [str(i) for i in range(1, NUM_CLASSES + 1)]  # ['1', ..., '9']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- MODEL -----------
class ChromiumCNN(nn.Module):
    def __init__(self, num_classes):
        super(ChromiumCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ----------- LOAD MODEL -----------
model = ChromiumCNN(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ----------- SELECT IMAGE -----------
Tk().withdraw()  # Hide the root window
file_path = filedialog.askopenfilename(
    title="Select Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)

if not file_path:
    print("❌ No image selected.")
    exit()

# ----------- PREPROCESS AND PREDICT -----------
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

image = Image.open(file_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = outputs.max(1)
    predicted_class = CLASS_NAMES[predicted.item()]

# ----------- OUTPUT -----------
print(f"\n✅ Selected file: {file_path}")
print(f"✅ Predicted Chromium Class: {predicted_class}\n")
