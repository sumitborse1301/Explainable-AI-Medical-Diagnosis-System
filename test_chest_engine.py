from PIL import Image
from vision.chest_xray_engine import ChestXrayEngine, overlay_cam

# 1. Load engine
engine = ChestXrayEngine(
    weight_path="vision/weights/densenet121_chestxray.pt"
)

# 2. Load a chest X-ray image
img = Image.open("test_xray.jpeg").convert("RGB")   # put a chest xray image here

# 3. Run prediction
result = engine.predict(img)

print("\n--- CHEST X-RAY RESULT ---")
print("Prediction:", result["prediction"])
print("Confidence:", result["confidence"])

print("\nTop 5 probabilities:")
for k, v in sorted(result["all_probs"].items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{k}: {v:.4f}")

# 4. Generate heatmap
overlay, raw = overlay_cam(img, result["cam"])
overlay.save("cam_overlay.png")
raw.save("cam_raw.png")

print("\nGrad-CAM images saved: cam_overlay.png, cam_raw.png")
