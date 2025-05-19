import torch
import cv2
import os
from collections import Counter

# --- CONFIGURATION ---
img_path = 'dataset/010.jpg'   # Update this to your image path
output_img_path = 'output/detected_shelf.jpg'
output_txt_path = 'output/detection_summary.txt'
confidence_threshold = 0.01    # You can increase for higher precision
max_display_width = 1280       # Auto-resize to fit screen width
grid_rows = 3
grid_cols = 5

# --- LOAD IMAGE ---
img = cv2.imread(img_path)
if img is None:
    print(f"❌ Error: Image not found at {img_path}")
    exit()

# --- LOAD MODEL ---
model = torch.hub.load('yolov5', 'yolov5s', source='local')
model.conf = confidence_threshold

# --- DETECTION ---
results = model(img)
df = results.pandas().xyxy[0]
product_count = len(df)
print(f"✅ Total products detected: {product_count}")

# --- DRAW DETECTIONS ---
for _, row in df.iterrows():
    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    label = f"{row['name']} {row['confidence']:.2f}"
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# --- COUNT BY CLASS ---
class_counts = Counter(df['name'])
summary_text = ' | '.join([f"{cls}({cnt})" for cls, cnt in class_counts.items()])

# --- OVERLAY INFO ---
cv2.rectangle(img, (0, 0), (max(500, len(summary_text)*7), 30), (0, 0, 0), -1)
cv2.putText(img, f"Detected: {product_count} | {summary_text}",
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

# --- OPTIONAL GRID OVERLAY ---
def draw_grid(image, rows=3, cols=5):
    h, w = image.shape[:2]
    for r in range(1, rows):
        y = int(h * r / rows)
        cv2.line(image, (0, y), (w, y), (200, 200, 200), 1)
    for c in range(1, cols):
        x = int(w * c / cols)
        cv2.line(image, (x, 0), (x, h), (200, 200, 200), 1)

draw_grid(img, rows=grid_rows, cols=grid_cols)

# --- RESIZE FOR SCREEN DISPLAY ---
h, w = img.shape[:2]
if w > max_display_width:
    scale = max_display_width / w
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

# --- SAVE OUTPUTS ---
os.makedirs('output', exist_ok=True)
cv2.imwrite(output_img_path, img)
with open(output_txt_path, 'w') as f:
    f.write(f"Total detected: {product_count}\n")
    for cls, cnt in class_counts.items():
        f.write(f"{cls}: {cnt}\n")

# --- DISPLAY IMAGE ---
cv2.namedWindow('Shelf Detection', cv2.WINDOW_NORMAL)
cv2.imshow('Shelf Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
