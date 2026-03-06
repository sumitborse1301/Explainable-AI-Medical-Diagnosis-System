import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
import torchxrayvision as xrv
import torchvision.transforms as transforms
from PIL import Image



# -----------------------------
# Chest X-ray Engine
# -----------------------------
class ChestXrayEngine:
    def __init__(self, weight_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load full pretrained chest X-ray model
        print("[INFO] Loading official TorchXRayVision chest model...")

        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.model = self.model.to(self.device)
        self.model.eval()

        self.labels = self.model.pathologies

        print("[INFO] Chest model loaded.")




        # Try to get pathology labels from model (TorchXRayVision style)
        if hasattr(self.model, "pathologies"):
            self.labels = self.model.pathologies
        else:
            # Fallback: create generic labels based on output size
            # (we will later map them to real chest pathologies)
            dummy = torch.zeros(1, 1, 224, 224).to(self.device)
            with torch.no_grad():
                out = self.model(dummy)
            num_classes = out.shape[1]
            self.labels = [f"Pathology_{i}" for i in range(num_classes)]



        # Hook for Grad-CAM
        self.gradients = None
        self.activations = None
        self.model.features[-1].register_forward_hook(self.save_activation)
        self.model.features[-1].register_backward_hook(self.save_gradient)



    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def save_activation(self, module, input, output):
        self.activations = output


    def preprocess(self, pil_img):
        img = pil_img.convert("L")
        img = img.resize((224, 224))

        img = np.array(img).astype(np.float32)
        img = img / 255.0

        img = np.expand_dims(img, axis=0)  # (1,H,W)
        img = np.expand_dims(img, axis=0)  # (1,1,H,W)

        img = torch.from_numpy(img).to(self.device)

        return img




    def predict(self, pil_img):
        x = self.preprocess(pil_img)
        x.requires_grad = True

        outputs = self.model(x)
        probs = torch.sigmoid(outputs)[0]

        top_idx = torch.argmax(probs)
        pred_label = self.labels[top_idx.item()]
        confidence = probs[top_idx].item()


        # Backprop for Grad-CAM
        self.model.zero_grad()
        outputs[0, top_idx].backward()

        cam = self.generate_cam(x)

        all_probs = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}

        return {
            "prediction": pred_label,
            "confidence": round(confidence, 4),
            "all_probs": all_probs,
            "cam": cam
        }

    def generate_cam(self, x):
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
# Utility: overlay heatmap
# -----------------------------
def overlay_cam(image: Image.Image, cam):
    img = cv2.resize(np.array(image), (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(overlay), Image.fromarray(heatmap)
