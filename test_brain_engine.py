from PIL import Image
from vision.brain_tumor_engine import BrainTumorEngine, overlay_brain_cam

engine = BrainTumorEngine("vision/weights/efficientnet_brain_tumor.pt")

img = Image.open("sample_brain.jpg")  # use any dataset image

result = engine.predict(img)

print("\n--- BRAIN TUMOR RESULT ---")
print("Prediction:", result["prediction"])
print("Confidence:", result["confidence"])
print("All probs:", result["all_probs"])

overlay, heatmap = overlay_brain_cam(img, result["cam"])
overlay.save("brain_cam_overlay.png")
heatmap.save("brain_cam_raw.png")

print("\nGrad-CAM saved: brain_cam_overlay.png , brain_cam_raw.png")
