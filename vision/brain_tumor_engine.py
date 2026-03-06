import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2


# MUST MATCH brain_dataset.py
BRAIN_CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]


# -----------------------------
# Brain Tumor Engine
# -----------------------------
class BrainTumorEngine:
    def __init__(self, weight_path):
        print("[INFO] Loading brain tumor model...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        self.model = models.efficientnet_b0(pretrained=False)
        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features, 4
        )

        state = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device)
        self.model.eval()

        # SAME transforms as training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])

        ])

        # Grad-CAM hooks
        self.gradients = None
        self.activations = None

        target_layer = self.model.features[-1]
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

        print("[INFO] Brain tumor model loaded.")


    # -----------------------------
    # Hooks
    # -----------------------------
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def save_activation(self, module, input, output):
        self.activations = output


    # -----------------------------
    # Preprocess
    # -----------------------------
    def preprocess(self, pil_img):
        if pil_img.mode != "L":
            pil_img = pil_img.convert("L")

        img = self.transform(pil_img)
        img = img.unsqueeze(0).to(self.device)
        return img


    # -----------------------------
    # Predict
    # -----------------------------
    def predict(self, pil_img):
        x = self.preprocess(pil_img)
        x.requires_grad = True

        outputs = self.model(x)
        probs = F.softmax(outputs, dim=1)[0]

        conf, idx = torch.max(probs, dim=0)
        pred_label = BRAIN_CLASSES[idx.item()]

        # Backprop for Grad-CAM
        self.model.zero_grad()
        outputs[0, idx].backward()

        cam = self.generate_cam()

        all_probs = {
            BRAIN_CLASSES[i]: float(probs[i].item())
            for i in range(len(BRAIN_CLASSES))
        }

        return {
            "prediction": pred_label,
            "confidence": round(float(conf.item()), 4),
            "all_probs": all_probs,
            "cam": cam
        }


    # -----------------------------
    # Grad-CAM generator
    # -----------------------------
    def generate_cam(self):
        grads = self.gradients[0].cpu().data.numpy()
        acts = self.activations[0].cpu().data.numpy()

        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam


# -----------------------------
# Overlay function
# -----------------------------
def overlay_brain_cam(pil_img, cam):
    img = np.array(pil_img.convert("RGB"))
    h, w, _ = img.shape

    cam = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return Image.fromarray(overlay), Image.fromarray(heatmap)
