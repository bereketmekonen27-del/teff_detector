import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class TeffClassifier:
    def __init__(self):
        self.class_names = ['head_smudge', 'healthy', 'pest_damage', 'teff_rust', 'wilting_moisture_stress']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Recreate model structure
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(self.class_names))
        
        # Find the weights file bundled next to this script
        current_dir = os.path.dirname(__file__)
        weights_path = os.path.join(current_dir, 'teff_disease_model.pth')
        
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        else:
            raise FileNotFoundError(f"Teff model weights missing at: {weights_path}")
            
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, pred_idx = torch.max(probabilities, 0)
            
        return {
            "diagnosis": self.class_names[pred_idx.item()],
            "confidence": confidence.item() * 100
        }
