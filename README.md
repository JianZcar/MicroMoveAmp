## How to Run the Project

### 1. **Clone the Repository**

```bash
git clone https://github.com/JianZcar/MicroMoveAmp.git
cd MicroMoveAmp
```

### 2. **Set up a Python Virtual Environment**

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. **Install Dependencies**

Make sure you have `pip` updated, then install required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*Note:*

* The `requirements.txt` file should include packages like `flask`, `opencv-python`, `opencv-contrib-python`, `numpy`, etc.
* If you don’t have `requirements.txt`, run:

```bash
pip install flask opencv-python opencv-contrib-python numpy
```

### 4. **Run the Flask Application**

```bash
export FLASK_APP=app.py        # On Windows (cmd): set FLASK_APP=app.py
export FLASK_ENV=development   # Optional: enables debug mode
flask run
```

The server will start, usually at `http://127.0.0.1:5000/`.

### 5. **Open the Web Interface**

Open your browser and navigate to:

```
http://127.0.0.1:5000/
```

You can now upload images or video frames and use the image processing features.

---

### Optional: Troubleshooting

* If you get errors related to `cv2.optflow`, ensure you installed `opencv-contrib-python`.
* If you have performance issues, try using lower-resolution images.
* For any module not found errors, double-check your virtual environment activation and installed packages.
