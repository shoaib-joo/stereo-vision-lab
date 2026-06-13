# Stereo Vision Lab

An interactive web app demonstrating stereo vision and epipolar geometry.

## Features
- **Stereo Matching** — SSD block-matching disparity maps
- **Feature Detection** — SIFT keypoints with ratio-test matching  
- **Fundamental Matrix** — Normalised 8-point algorithm
- **Epipolar Lines** — Visualise epipolar constraints on both images
- **Rectification** — Uncalibrated stereo rectification

## Run Locally

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:5050
```

## Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app --workers 2 --timeout 120 --bind 0.0.0.0:$PORT`
   - **Environment variable:** `SECRET_KEY` = (any long random string)
5. Deploy — you'll get a public URL in ~2 minutes

## Stack
- Python · Flask · OpenCV · NumPy
- Deployed via Render (free tier)
