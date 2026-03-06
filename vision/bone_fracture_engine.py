import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Grad-CAM helper
# -------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, class_idx):
        grads = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (grads * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


# -------------------------
# Bone fracture engine
# -------------------------
class BoneFractureEngine:
    def __init__(self, weight_path):
        print("[INFO] Loading bone fracture model...")

        self.model = models.efficientnet_b0(pretrained=False)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)

        state = torch.load(weight_path, map_location=DEVICE)
        self.model.load_state_dict(state)

        self.model.to(DEVICE)
        self.model.eval()

        # last conv block for GradCAM
        self.target_layer = self.model.features[-1]
        self.gradcam = GradCAM(self.model, self.target_layer)

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

        print("[INFO] Bone fracture model loaded.")

    # -------------------------
    # Preprocess
    # -------------------------
    def preprocess(self, pil_img):
        img = pil_img.convert("L")
        x = self.transform(img).unsqueeze(0).to(DEVICE)
        return x

    # -------------------------
    # Prediction
    # -------------------------
    def predict(self, pil_img):
        x = self.preprocess(pil_img)
        x.requires_grad = True

        outputs = self.model(x)
        probs = torch.softmax(outputs, dim=1)[0]

        conf, pred_idx = torch.max(probs, dim=0)

        labels = ["Normal", "Fracture"]
        prediction = labels[pred_idx.item()]

        # Grad-CAM
        self.model.zero_grad()
        outputs[0, pred_idx].backward()
        cam = self.gradcam.generate(pred_idx)

        cam = cam[0,0].cpu().numpy()
        cam = cv2.resize(cam, pil_img.size)

        all_probs = {
            "Normal": float(probs[0].item()),
            "Fracture": float(probs[1].item())
        }

        return {
            "prediction": prediction,
            "confidence": float(conf.item()),
            "all_probs": all_probs,
            "cam": cam
        }


# -------------------------
# Heatmap overlay helper
# -------------------------
def overlay_cam(pil_img, cam):
    img = np.array(pil_img.convert("RGB"))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return Image.fromarray(overlay), Image.fromarray(heatmap)
