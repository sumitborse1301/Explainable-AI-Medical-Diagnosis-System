from PIL import Image
from vision.chest_multidisease_engine import ChestMultiDiseaseEngine, overlay_chest_cam

engine = ChestMultiDiseaseEngine("vision/weights/chest_multidisease_tb.pt")

img = Image.open("sample_tb.png")

result = engine.predict(img)

print("\n--- CHEST RESULT ---")
print("Prediction:", result["prediction"])
print("Confidence:", result["confidence"])
print("\nTop probabilities:")
for k, v in sorted(result["all_probs"].items(), key=lambda x: x[1], reverse=True)[:8]:
    print(f"{k}: {v:.4f}")

overlay, heatmap = overlay_chest_cam(img, result["cam"])
overlay.save("chest_cam_overlay.png")
heatmap.save("chest_cam_raw.png")

print("\nGrad-CAM saved: chest_cam_overlay.png , chest_cam_raw.png")
