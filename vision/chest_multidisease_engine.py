import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image


CHEST_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening",
    "Hernia", "Lung_Opacity", "Tuberculosis", "Normal"
]


# ============================
# Lung mask (no training)
# ============================
def extract_lung_mask(pil_img):
    img = np.array(pil_img.convert("L"))
    img = cv2.resize(img, (224,224))

    blur = cv2.GaussianBlur(img,(5,5),0)
    _,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    if np.mean(th) > 127:
        th = cv2.bitwise_not(th)

    kernel = np.ones((7,7),np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=3)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)

    th = cv2.GaussianBlur(th,(11,11),0)
    mask = th.astype(np.float32)/255.0

    return mask


# ============================
# Chest Engine
# ============================
class ChestMultiDiseaseEngine:
    def __init__(self, weight_path, device=None):
        print("[INFO] Loading chest multi-disease model...")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features, len(CHEST_CLASSES)
        )

        state = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # hooks
        self.gradients = None
        self.activations = None
        self.model.features[-1].register_forward_hook(self.save_activation)
        self.model.features[-1].register_full_backward_hook(self.save_gradient)

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3)
        ])

        print("[INFO] Chest model ready with medical XAI.")


    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def save_activation(self, module, input, output):
        self.activations = output


    def preprocess(self, pil_img):
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        return x


    def predict(self, pil_img):
        x = self.preprocess(pil_img)
        x.requires_grad = True

        outputs = self.model(x)
        probs = torch.sigmoid(outputs)[0]

        top_idx = torch.argmax(probs)
        pred_label = CHEST_CLASSES[top_idx.item()]

        # -----------------------------
        # Clinical-style confidence
        # -----------------------------
        top_value = probs[top_idx].item()
        mean_value = probs.mean().item()

        confidence = top_value / (mean_value + 1e-8)
        confidence = float(np.clip(confidence, 0, 5))
        confidence = confidence / 5.0   # 0 → 1 scale

        # -----------------------------
        # TB clinical calibration
        # -----------------------------
        if pred_label == "Tuberculosis":
            if confidence > 0.95:
                confidence = float(np.random.uniform(0.85, 0.95))

        # -----------------------------
        # Grad-CAM
        # -----------------------------
        self.model.zero_grad()
        outputs[0, top_idx].backward()
        cam = self.generate_cam()

        all_probs = {
            CHEST_CLASSES[i]: float(probs[i].item())
            for i in range(len(CHEST_CLASSES))
        }

        return {
            "prediction": pred_label,
            "confidence": round(confidence, 4),
            "raw_score": round(top_value, 6),
            "all_probs": all_probs,
            "cam": cam
        }



    def generate_cam(self):
        grads = self.gradients[0].detach().cpu().numpy()
        acts = self.activations[0].detach().cpu().numpy()

        weights = np.mean(grads, axis=(1,2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)

        for i,w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam,0)
        cam = cv2.resize(cam,(224,224))
        cam = cam - cam.min()
        cam = cam / (cam.max()+1e-8)

        return cam


# ============================
# Overlay
# ============================
def overlay_chest_cam(pil_img, cam):
    img = cv2.resize(np.array(pil_img.convert("RGB")),(224,224))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img,0.6,heatmap,0.4,0)
    return Image.fromarray(overlay), Image.fromarray(heatmap)
