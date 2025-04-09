import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import open_clip
from sklearn.metrics.pairwise import cosine_similarity
import heapq

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
yolo_model = YOLO("yolov8m.pt")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)
clip_model = clip_model.to(device).eval()

def video_to_frames_with_timestamp(video_path, output_folder, every_n_frames, stream_name="stream"):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % every_n_frames == 0:
            timestamp = frame_count / fps
            filename = f"{stream_name}_frame{saved_count:05d}.jpg"
            frame_path = os.path.join(output_folder, filename)
            cv2.imwrite(frame_path, frame)
            with open(frame_path + ".txt", 'w') as f:
                f.write(str(timestamp))
            saved_count += 1
        frame_count += 1

    cap.release()

def get_clip_embedding(image_array):
    image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.squeeze().cpu().numpy()

def process_frames_with_timestamps(frames_folder, base_results_path):
    annotated_folder = os.path.join(base_results_path, "annotated_frames")
    cropped_folder = os.path.join(base_results_path, "cropped_objects")

    os.makedirs(annotated_folder, exist_ok=True)
    os.makedirs(cropped_folder, exist_ok=True)

    results = []
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
    for fname in frame_files:
        frame_path = os.path.join(frames_folder, fname)
        timestamp_path = frame_path + ".txt"
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        try:
            with open(timestamp_path, 'r') as f:
                timestamp = float(f.read())
        except FileNotFoundError:
            timestamp = None

        display_frame = frame.copy()
        detections = yolo_model.predict(frame, verbose=False)[0]

        for i, box in enumerate(detections.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            embedding = get_clip_embedding(cropped)
            crop_filename = f"{fname[:-4]}_obj{i}.jpg"
            crop_path = os.path.join(cropped_folder, crop_filename)
            cv2.imwrite(crop_path, cropped)

            results.append({
                "frame": fname,
                "timestamp": timestamp,
                "bbox": [x1, y1, x2, y2],
                "confidence": float(box.conf[0]),
                "embedding": embedding.tolist(),
                "cropped_image_path": crop_path
            })

            label = f"{box.conf[0]:.2f}"
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        annotated_path = os.path.join(annotated_folder, fname)
        cv2.imwrite(annotated_path, display_frame)

    return results


def search_similar_objects(query_image_path, results, top_k=3):
    img = Image.open(query_image_path).convert("RGB")
    img_array = np.array(img)
    query_embedding = get_clip_embedding(img_array)

    similarities = []
    for entry in results:
        db_embedding = np.array(entry["embedding"])
        sim = cosine_similarity([query_embedding], [db_embedding])[0][0]
        similarities.append((sim, {
            "frame": entry["frame"],
            "timestamp": entry["timestamp"],
            "bbox": entry["bbox"],
            "confidence": entry["confidence"],
            "similarity": sim
        }))

    top_matches = heapq.nlargest(top_k, similarities, key=lambda x: x[0])
    return [match[1] for match in top_matches]

def show_matches(top_results, frames_folder="output_frames", output_folder="final_results"):
    saved_files = []
    for i, res in enumerate(top_results):
        path = os.path.join(frames_folder, res["frame"])
        frame = cv2.imread(path)
        if frame is None:
            continue
        x1, y1, x2, y2 = map(int, res["bbox"])
        cropped = frame[y1:y2, x1:x2]
        output_path = os.path.join(output_folder, f"match_{i+1}_{res['frame']}")
        cv2.imwrite(output_path, cropped)
        saved_files.append(os.path.basename(output_path))
    return saved_files

def run_pipeline(video1_path, video2_path, image_path):
    frames_folder = "output_frames"
    os.makedirs(frames_folder, exist_ok=True)

    video_to_frames_with_timestamp(video1_path, frames_folder, every_n_frames=2, stream_name="video1")
    video_to_frames_with_timestamp(video2_path, frames_folder, every_n_frames=2, stream_name="video2")

    results = process_frames_with_timestamps(frames_folder)
    top_results = search_similar_objects(image_path, results)
    result_files = show_matches(top_results, frames_folder=frames_folder)
    return result_files
