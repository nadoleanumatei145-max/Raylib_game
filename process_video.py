#!/usr/bin/env python3
"""
Depth Anything V2 - Video Processing
Procesează un videoclip și generează depth map pentru fiecare cadru
"""

import cv2
import numpy as np
import onnxruntime as ort
import sys
import os
import time
from tqdm import tqdm

def preprocess_image(frame, target_size=518):
    """Preprocesare pentru Depth Anything V2"""
    h, w = frame.shape[:2]
    
    # Redimensionează păstrând proporțiile
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad la pătrat
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    
    # Normalizează [0,1] și convertește la [C,H,W]
    img = canvas.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    return img, (h, w), (new_h, new_w)

def postprocess_depth(output, orig_shape, crop_shape, target_size=518):
    """Postprocesare depth map"""
    # Adâncimea e pe canalul 1
    depth = output[0, 0]  # shape: (H, W)
    
    # Normalizează la 0-255
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth = (depth * 255).astype(np.uint8)
    
    # Taie padding-ul
    h, w = crop_shape
    depth = depth[:h, :w]
    
    # Redimensionează la dimensiunea originală
    orig_h, orig_w = orig_shape
    depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
    return depth

def enhance_depth_for_anime(depth, original_frame):
    """Îmbunătățește depth map-ul pentru anime"""
    try:
        # Detectează margini
        gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilată marginile
        kernel = np.ones((3,3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Aplică filtru median
        depth_blurred = cv2.medianBlur(depth, 3)
        
        # La margini, păstrează contrastul
        depth_enhanced = depth_blurred.copy()
        depth_enhanced[edges_dilated > 0] = depth[edges_dilated > 0]
        
        # Crește contrastul general
        depth_enhanced = cv2.equalizeHist(depth_enhanced)
        
        return depth_enhanced
    except Exception as e:
        print(f"⚠️ Eroare la îmbunătățire: {e}")
        return depth

def process_video(input_path, output_path, model_path="depth_anything_v2_small.onnx", 
                  target_size=518, enhance=True, save_frames=False):
    """Procesează videoclipul frame cu frame"""
    
    # Verifică dacă modelul există
    if not os.path.exists(model_path):
        print(f"❌ Modelul {model_path} nu există!")
        print("📥 Descarcă modelul de la:")
        print("   https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_small.onnx")
        sys.exit(1)
    
    # Verifică dacă videoclipul există
    if not os.path.exists(input_path):
        print(f"❌ Videoclipul {input_path} nu există!")
        sys.exit(1)
    
    # Încarcă modelul ONNX
    print("🧠 Se încarcă modelul Depth Anything V2...")
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print("✅ Model încărcat!")
    except Exception as e:
        print(f"❌ Eroare la încărcare: {e}")
        sys.exit(1)
    
    # Deschide videoclipul
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Nu pot deschide videoclipul: {input_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height}, {fps:.2f} fps, {total_frames} cadre")
    
    # Pregătește writer-ul pentru output
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), False)
    
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), False)
        if not out.isOpened():
            print("❌ Nu pot crea fișierul de ieșire")
            return
    
    # Crează folder pentru frame-uri (opțional)
    if save_frames:
        frames_dir = "frames_output"
        os.makedirs(frames_dir, exist_ok=True)
    
    print("🎬 Se procesează...")
    start_time = time.time()
    
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # Preprocesare
            input_blob, orig_shape, crop_shape = preprocess_image(frame, target_size)
            
            # Inferență
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            outputs = session.run([output_name], {input_name: input_blob.astype(np.float32)})
            
            # Postprocesare
            depth = postprocess_depth(outputs[0], orig_shape, crop_shape, target_size)
            
            # Îmbunătățire pentru anime
            if enhance:
                depth = enhance_depth_for_anime(depth, frame)
            
            # Scrie în output
            out.write(depth)
            
            # Salvează frame-uri individuale (opțional)
            if save_frames:
                frame_path = os.path.join(frames_dir, f"frame_{i:06d}.png")
                cv2.imwrite(frame_path, depth)
            
        except Exception as e:
            print(f"⚠️ Eroare la cadrul {i}: {e}")
            continue
    
    cap.release()
    out.release()
    
    elapsed = time.time() - start_time
    print(f"\n✅ Finalizat! {total_frames} cadre în {elapsed:.1f} secunde")
    print(f"📁 Output: {output_path}")
    print(f"📊 Viteză: {total_frames/elapsed:.1f} fps")

def main():
    if len(sys.argv) < 3:
        print("📖 Utilizare: python process_video.py <input_video> <output_video>")
        print("Exemplu: python process_video.py anime.mp4 depth_output.mp4")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Parametri opționali
    target_size = 518  # 518 e standard pentru Depth Anything V2
    enhance = True
    
    print("🚀 Depth Anything V2 - Video Depth Map Generator")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Target size: {target_size}")
    print(f"   Anime enhancement: {enhance}")
    print("=" * 50)
    
    process_video(input_path, output_path, target_size=target_size, enhance=enhance)

if __name__ == "__main__":
    main()
