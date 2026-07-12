import os
import time
import shutil
import numpy as np
from PIL import Image
import onnxruntime as ort

# Configuration
ONNX_PATH = "jzsz_classifier_squeezenet.onnx"
FRAMES_DIR = "frames"
NEW_DATA_DIR = "new_data"
POSITIVE_DIR = "positive_candidates"
IMG_SIZE = 224

# Load ONNX model
session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# Image preprocessing constants
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(arr, axis=0).astype(np.float32)

def predict(image_path):
    input_tensor = preprocess(image_path)
    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    return probs[1], int(np.argmax(logits))  # prob of positive class, predicted class

# Benchmark single inference speed
print("Benchmarking inference speed...")
# Find an existing test directory for benchmark
TEST_DIRS = [NEW_DATA_DIR, POSITIVE_DIR]
bench_dir = None
for d in TEST_DIRS:
    if os.path.exists(d) and os.listdir(d):
        bench_dir = d
        break

if bench_dir is None:
    raise FileNotFoundError("No test directory found")

img_path = os.path.join(bench_dir, sorted(os.listdir(bench_dir))[0])
input_tensor = preprocess(img_path)

# Warmup
for _ in range(10):
    session.run(None, {input_name: input_tensor})

n = 1000
start = time.perf_counter()
for _ in range(n):
    session.run(None, {input_name: input_tensor})
end = time.perf_counter()

elapsed = end - start
fps = n / elapsed
ms_per_frame = elapsed / n * 1000
print(f"ONNX CPU inference: {n} frames in {elapsed:.3f}s => {fps:.1f} FPS, {ms_per_frame:.2f} ms/frame")

# Run on all test frames (frames/ + new_data/)
print("\nRunning on all frames...")
results = []
for src_dir in [FRAMES_DIR, NEW_DATA_DIR]:
    if not os.path.exists(src_dir):
        continue
    files = sorted([f for f in os.listdir(src_dir) if f.endswith('.jpg')])
    for f in files:
        path = os.path.join(src_dir, f)
        prob, pred = predict(path)
        results.append((f, prob, pred, src_dir))

results.sort(key=lambda x: x[1], reverse=True)

print(f"Total frames: {len(results)}")
print(f"Predicted positive: {sum(1 for _, _, pred, _ in results if pred == 1)}")
print(f"Predicted negative: {sum(1 for _, _, pred, _ in results if pred == 0)}")
print('\nTop 20 positive probabilities:')
for name, prob, pred, src_dir in results[:20]:
    print(f'  [{src_dir}] {name}: {prob:.3f} (pred={pred})')

print('\nBottom 20 positive probabilities:')
for name, prob, pred, src_dir in results[-20:]:
    print(f'  [{src_dir}] {name}: {prob:.3f} (pred={pred})')

# Direct evaluation on positive_candidates files
print(f"\nDirect evaluation on positive_candidates/:")
positive_files = sorted([f for f in os.listdir(POSITIVE_DIR) if f.endswith('.jpg')])
positive_detected = 0
undetected = []
for f in positive_files:
    path = os.path.join(POSITIVE_DIR, f)
    prob, pred = predict(path)
    if pred == 1:
        positive_detected += 1
    else:
        undetected.append((f, prob))
print(f"Positive samples detected as positive: {positive_detected}/{len(positive_files)} ({100*positive_detected/len(positive_files):.1f}%)")
if undetected:
    print('Undetected positives:')
    for f, prob in undetected:
        print(f'  {f}: prob={prob:.3f}')

# Save positive predictions to folder
pred_dir = "predicted_positive"
if os.path.exists(pred_dir):
    shutil.rmtree(pred_dir)
os.makedirs(pred_dir, exist_ok=True)
for f, prob, pred, src_dir in results:
    if pred == 1:
        src = os.path.join(src_dir, f)
        dst = os.path.join(pred_dir, f'{f[:-4]}_prob{prob:.3f}.jpg')
        with open(src, 'rb') as src_file, open(dst, 'wb') as dst_file:
            dst_file.write(src_file.read())
print(f"\nSaved {sum(1 for _, _, pred, _ in results if pred == 1)} predicted positive frames to {pred_dir}/")
