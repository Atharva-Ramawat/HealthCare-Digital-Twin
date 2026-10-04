import os
import sys

# 1. Add project root to Python path to resolve 'ml' module
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from ml.cxr.model import DenseNet121Pulmonary
from ml.cxr.config import CXRConfig

# 2. Load canonical weights
config = CXRConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNet121Pulmonary(num_classes=8, pretrained=False).to(device)

checkpoint_path = os.path.join(config.checkpoints_dir, "densenet121_mimic_cxr.pt")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# 3. Preprocess a sample test image
# (Make sure this path points to a valid image in your data/raw folder)
img_path = "data/raw/mimic_cxr_aug_validate/files/p10/p10003502/s50084553/70d7e600-373c1311-929f5ff9-23ee3621-ff551ff9.jpg"
img = Image.open(img_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img).unsqueeze(0).to(device)

# 4. Predict Multi-Label Probabilities
with torch.no_grad():
    probs = model.predict_probabilities(img_tensor).cpu().numpy()[0]
    
print("--- Predicted Probabilities ---")
for cls, prob in zip(config.target_classes, probs):
    print(f"{cls}: {prob:.4f}")

# 5. Generate Grad-CAM for the highest-scoring pathology
best_class_idx = probs.argmax()
heatmap = model.generate_gradcam_heatmap(img_tensor, class_idx=best_class_idx, target_size=(224, 224))

# 6. Save the visual output
plt.imshow(img.resize((224, 224)))
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.title(f"Grad-CAM: {config.target_classes[best_class_idx]}")
plt.savefig("gradcam_test_output.png")
print("Saved Grad-CAM visualization to gradcam_test_output.png")