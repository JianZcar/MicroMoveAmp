from flask import Flask, render_template, request, jsonify
from io import BytesIO
from processor import process_frames
import base64
import cv2
import numpy as np

from processor import process_frames  # Your processing function, updated to accept and return numpy arrays

app = Flask(__name__)


def read_images(files):
    images = []
    for file in files:
        in_memory_file = file.read()
        nparr = np.frombuffer(in_memory_file, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("One of the files is not a valid image")
        images.append(img)
    return images


@app.route("/")
def index():
    # Render single-page app
    return render_template("index.html")


@app.route('/process', methods=['POST'])
def process():
    files = request.files.getlist('frames')
    if len(files) < 4:
        return jsonify({"error": "Please upload at least 4 images"}), 400

    frames = []
    for f in files:
        # Read file bytes, convert to numpy array, decode image in grayscale
        file_bytes = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({"error": "One or more files could not be decoded as images"}), 400
        frames.append(img)

    # Run your existing processing function (should accept list of cv2 grayscale images)
    # Make sure your process_frames returns a single grayscale or BGR image as np.ndarray
    output_img = process_frames(frames, out_vis_path=None)  # out_vis_path=None means no save to disk

    # Encode output image to PNG in memory
    success, buffer = cv2.imencode('.png', output_img)
    if not success:
        return jsonify({"error": "Failed to encode output image"}), 500

    # Base64 encode bytes for JSON transport
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({"image": f"data:image/png;base64,{img_base64}"})


if __name__ == "__main__":
    app.run(debug=True)

