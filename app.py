from flask import Flask, render_template, request, redirect, send_from_directory
import os
import shutil
import time
from werkzeug.utils import secure_filename

from main import (
    video_to_frames_with_timestamp,
    process_frames_with_timestamps,
    search_similar_objects,
    show_matches
)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Setup runtime folder structure
    timestamp = str(int(time.time()))
    base_results_path = os.path.join("results")

    UPLOAD_FOLDER = os.path.join(base_results_path, "inputs")
    FRAMES_FOLDER = os.path.join(base_results_path, "output_frames")
    RESULTS_FOLDER = os.path.join(base_results_path, "final_results")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(FRAMES_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # Save uploaded files
    video1 = request.files['video1']
    video2 = request.files['video2']
    image = request.files['image']

    video1_filename = f"{timestamp}_{secure_filename(video1.filename)}"
    video2_filename = f"{timestamp}_{secure_filename(video2.filename)}"
    image_filename = f"{timestamp}_{secure_filename(image.filename)}"

    video1_path = os.path.join(UPLOAD_FOLDER, video1_filename)
    video2_path = os.path.join(UPLOAD_FOLDER, video2_filename)
    image_path = os.path.join(UPLOAD_FOLDER, image_filename)

    video1.save(video1_path)
    video2.save(video2_path)
    image.save(image_path)

    # Run pipeline
    video_to_frames_with_timestamp(video1_path, FRAMES_FOLDER, every_n_frames=5, stream_name="video1")
    video_to_frames_with_timestamp(video2_path, FRAMES_FOLDER, every_n_frames=5, stream_name="video2")
    results = process_frames_with_timestamps(FRAMES_FOLDER, base_results_path)
    top_results = search_similar_objects(image_path, results)
    show_matches(top_results, frames_folder=FRAMES_FOLDER, output_folder=RESULTS_FOLDER)

    # Save path for serving
    global last_results_path
    last_results_path = RESULTS_FOLDER

    return redirect('/results')

@app.route('/results')
def results():
    images = os.listdir(last_results_path)
    images = [img for img in images if img.endswith(('.jpg', '.png'))]
    return render_template('results.html', images=images)

@app.route('/final_results/<filename>')
def send_image(filename):
    return send_from_directory(last_results_path, filename)

@app.route('/cleanup')
def cleanup():
    shutil.rmtree("results", ignore_errors=True)
    return redirect('/')

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


