#!/usr/bin/env python3
import cv2
import numpy as np
import onnxruntime as ort
import sys
import os
from tqdm import tqdm

def preprocess_image(frame, target_size=518):
    h, w = frame.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    img = canvas.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, (h, w), (new_h, new_w)

def postprocess_depth(output, orig_shape, crop_shape):
    depth = output[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth = (depth * 255).astype(np.uint8)
    h, w = crop_shape
    depth = depth[:h, :w]
    orig_h, orig_w = orig_shape
    depth = cv2.resize(depth, (orig_w, orig_h))
    return depth

def process_video(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"❌ Videoclipul {input_path} nu există!")
        sys.exit(1)
    
    print("🧠 Se încarcă modelul...")
    session = ort.InferenceSession("depth_anything_v2_small.onnx", providers=['CPUExecutionProvider'])
    
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 {w}x{h}, {fps:.2f} fps, {total} cadre")
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), False)
    
    for i in tqdm(range(total)):
        ret, frame = cap.read()
        if not ret:
            break
        input_blob, orig_shape, crop_shape = preprocess_image(frame)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        outputs = session.run([output_name], {input_name: input_blob.astype(np.float32)})
        depth = postprocess_depth(outputs[0], orig_shape, crop_shape)
        out.write(depth)
    
    cap.release()
    out.release()
    print(f"✅ Finalizat! Output: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_video.py input.mp4 output.mp4")
        sys.exit(1)
    process_video(sys.argv[1], sys.argv[2])
