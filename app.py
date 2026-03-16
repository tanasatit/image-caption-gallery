"""
Image Caption Gallery - A Web-Based Personal Gallery with ML Image Captioning
Uses Salesforce/blip-image-captioning-base from Hugging Face for AI-generated captions.

Features:
- Create and manage photo albums
- Upload multiple images to albums
- Browse images in a gallery view
- AI-powered image captioning using BLIP model (via Hugging Face Inference API)
- Responsive, modern UI

To run:
    pip install flask requests pillow
    python app.py
    Open http://localhost:5000
"""

from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
import uuid
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image
import logging
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'gallery-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Load BLIP model locally (downloaded once and cached by transformers)
logger_init = logging.getLogger(__name__ + ".init")
logger_init.info("Loading BLIP model (first run will download ~900 MB)...")
_blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
_blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
_blip_model.eval()
_device = "cuda" if torch.cuda.is_available() else "cpu"
_blip_model.to(_device)
logger_init.info(f"BLIP model loaded on {_device}.")

# Simple JSON-based storage (no database needed)
DATA_FILE = "gallery_data.json"

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def load_data():
    """Load gallery data from JSON file."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"albums": [], "images": []}


def save_data(data):
    """Save gallery data to JSON file."""
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# ML Component – BLIP Image Captioning via Hugging Face Inference API
# ---------------------------------------------------------------------------

def generate_caption(image_path):
    """
    Generate a caption locally using Salesforce/blip-image-captioning-base
    via the transformers library (no external API call).
    """
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = _blip_processor(images=image, return_tensors="pt").to(_device)
        with torch.no_grad():
            output_ids = _blip_model.generate(**inputs, max_new_tokens=50)
        caption = _blip_processor.decode(output_ids[0], skip_special_tokens=True)
        logger.info(f"Caption generated: {caption}")
        return caption
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        return f"Error generating caption: {str(e)}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    """Home page – show all albums."""
    data = load_data()
    # Count images per album
    for album in data['albums']:
        album['image_count'] = sum(1 for img in data['images'] if img['album_id'] == album['id'])
        # Get cover image (first image in album)
        cover = next((img for img in data['images'] if img['album_id'] == album['id']), None)
        album['cover'] = cover['filename'] if cover else None
    return render_template('index.html', albums=data['albums'])


@app.route('/album/create', methods=['POST'])
def create_album():
    """Create a new album."""
    name = request.form.get('name', '').strip()
    description = request.form.get('description', '').strip()
    if not name:
        return jsonify({'error': 'Album name is required'}), 400

    data = load_data()
    album = {
        'id': str(uuid.uuid4())[:8],
        'name': name,
        'description': description,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M')
    }
    data['albums'].append(album)
    save_data(data)
    return jsonify({'success': True, 'album': album})


@app.route('/album/<album_id>')
def view_album(album_id):
    """View images in an album."""
    data = load_data()
    album = next((a for a in data['albums'] if a['id'] == album_id), None)
    if not album:
        return redirect(url_for('index'))
    images = [img for img in data['images'] if img['album_id'] == album_id]
    return render_template('album.html', album=album, images=images)


@app.route('/album/<album_id>/delete', methods=['POST'])
def delete_album(album_id):
    """Delete an album and all its images."""
    data = load_data()
    # Remove image files
    for img in data['images']:
        if img['album_id'] == album_id:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], img['filename'])
            if os.path.exists(filepath):
                os.remove(filepath)
    data['images'] = [img for img in data['images'] if img['album_id'] != album_id]
    data['albums'] = [a for a in data['albums'] if a['id'] != album_id]
    save_data(data)
    return jsonify({'success': True})


@app.route('/album/<album_id>/upload', methods=['POST'])
def upload_images(album_id):
    """Upload one or more images to an album."""
    data = load_data()
    album = next((a for a in data['albums'] if a['id'] == album_id), None)
    if not album:
        return jsonify({'error': 'Album not found'}), 404

    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400

    uploaded = []
    for file in files:
        if file and allowed_file(file.filename):
            # Generate unique filename
            ext = file.filename.rsplit('.', 1)[1].lower()
            unique_name = f"{uuid.uuid4().hex[:12]}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            file.save(filepath)

            # Generate AI caption
            caption = generate_caption(filepath)

            image_record = {
                'id': str(uuid.uuid4())[:8],
                'album_id': album_id,
                'filename': unique_name,
                'original_name': file.filename,
                'caption': caption,
                'uploaded_at': datetime.now().strftime('%Y-%m-%d %H:%M')
            }
            data['images'].append(image_record)
            uploaded.append(image_record)

    save_data(data)
    return jsonify({'success': True, 'images': uploaded})


@app.route('/image/<image_id>/caption', methods=['POST'])
def regenerate_caption(image_id):
    """Regenerate the caption for an image."""
    data = load_data()
    image = next((img for img in data['images'] if img['id'] == image_id), None)
    if not image:
        return jsonify({'error': 'Image not found'}), 404

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], image['filename'])
    caption = generate_caption(filepath)
    image['caption'] = caption
    save_data(data)
    return jsonify({'success': True, 'caption': caption})


@app.route('/image/<image_id>/delete', methods=['POST'])
def delete_image(image_id):
    """Delete a single image."""
    data = load_data()
    image = next((img for img in data['images'] if img['id'] == image_id), None)
    if not image:
        return jsonify({'error': 'Image not found'}), 404

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], image['filename'])
    if os.path.exists(filepath):
        os.remove(filepath)
    data['images'] = [img for img in data['images'] if img['id'] != image_id]
    save_data(data)
    return jsonify({'success': True})


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("  Image Caption Gallery")
    print("  Using: Salesforce/blip-image-captioning-base")
    print("  Open: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)