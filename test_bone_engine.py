from PIL import Image
from vision.bone_fracture_engine import BoneFractureEngine, overlay_cam

engine = BoneFractureEngine(
    weight_path="vision/weights/efficientnet_bone_fracture.pt"
)

img = Image.open("C:/Users/sumit/OneDrive/Pictures/XAI Doctor Diagnosis Project Currently Working/data/mura/train/XR_WRIST/patient00006/study1_positive/image1.png")  # put any MURA image here

result = engine.predict(img)

print("\n--- BONE FRACTURE RESULT ---")
print("Prediction:", result["prediction"])
print("Confidence:", result["confidence"])
print("All probs:", result["all_probs"])

overlay, raw = overlay_cam(img, result["cam"])
overlay.save("bone_cam_overlay.png")
raw.save("bone_cam_raw.png")

print("\nGrad-CAM saved: bone_cam_overlay.png , bone_cam_raw.png")
