# Stereo Vision Lab

An interactive web app for exploring stereo vision and epipolar geometry — built with FastAPI and OpenCV.

**🔗 Live demo: [stereo-vision-lab.onrender.com](https://stereo-vision-lab.onrender.com/)**

> Free-tier hosting — the app may take 30–60s to wake up on first visit.

---

## What it does

Upload a stereo image pair (or use the included demo images) and walk through the classic stereo vision pipeline, with every step visualised directly in the browser:

| Step | What happens |
|---|---|
| **Stereo Matching** | Dense disparity map via SSD block matching |
| **Feature Detection** | SIFT keypoints matched with a ratio test |
| **Fundamental Matrix** | Estimated via the normalised 8-point algorithm |
| **Epipolar Geometry** | Computes epipoles and draws epipolar lines |
| **Rectification** | Uncalibrated rectification + re-computed disparity |

## Tech Stack

- **Backend:** FastAPI, Uvicorn
- **Computer Vision:** OpenCV, NumPy
- **Frontend:** Jinja2, HTML/CSS/JS

## Running Locally

```bash
git clone https://github.com/shoaib-joo/stereo-vision-lab
cd stereo_app
pip install -r requirements.txt
cp .env.example .env   # add your own SECRET_KEY
python app.py
```

Open [http://localhost:5050](http://localhost:5050).

## Project Structure

```
stereo_app/
├── app.py            # FastAPI routes
├── stereo_core.py    # Stereo vision & geometry algorithms
├── templates/        # HTML templates
├── static/           # Default sample images & assets
├── requirements.txt
└── Procfile
```

## License

For educational and demonstration purposes.