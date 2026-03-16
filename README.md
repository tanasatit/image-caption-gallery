# Image Caption Gallery

A web-based personal gallery application enhanced with ML-powered image captioning using the **Salesforce/blip-image-captioning-base** model from Hugging Face.

## Features

- **Album Management** — Create, browse, and delete photo albums
- **Multi-Image Upload** — Upload multiple images at once via click or drag-and-drop
- **AI Image Captioning** — Automatic captions generated using BLIP (Bootstrapping Language-Image Pre-training) model via Hugging Face Inference API
- **Caption Regeneration** — Re-generate captions for any image with one click
- **Lightbox View** — Click any image to view it full-size with its caption
- **Responsive Design** — Works on desktop and mobile
- **No Database Required** — Uses simple JSON file storage

## Tech Stack

- **Backend:** Python / Flask
- **Frontend:** HTML, CSS, JavaScript (vanilla)
- **ML Model:** [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
- **API:** Hugging Face Inference API

## How the ML Component Works

The application calls the Hugging Face Inference API with each uploaded image. The BLIP model processes the image and returns a natural language caption describing what it sees. This runs on Hugging Face's servers, so no local GPU is needed.

```
User uploads image → Flask backend → Hugging Face API (BLIP model) → Caption returned → Displayed in gallery
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd image-caption-gallery
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Set Hugging Face API token for higher rate limits:**
   ```bash
   export HF_API_TOKEN=your_token_here
   ```
   The app works without a token, but may hit rate limits with heavy use.

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Open in browser:**
   ```
   http://localhost:5000
   ```

## Project Structure

```
image-caption-gallery/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── gallery_data.json      # Auto-created data storage
├── static/
│   └── uploads/           # Uploaded images stored here
└── templates/
    ├── index.html         # Home page (album listing)
    └── album.html         # Album view (image grid + upload)
```

## Prediction Samples

| Image | AI-Generated Caption |
|-------|---------------------|
| (tennis photo) | "two women shaking hands on a tennis court with a crowd of people in the background" |
| (your sample) | (paste caption here) |

## Activity 2.5 Requirements Met

- **(1) Web-Based Personal Gallery** — Users can browse and upload multiple images to create albums
- **(2) ML Image Captioning** — Integrated BLIP model from Hugging Face to generate captions while browsing images
- **Python + Flask** — Built with Flask as suggested
- **GitHub Submission** — Code and prediction samples included

## Author

Tanasatit Ngaosupathon 6610545804
