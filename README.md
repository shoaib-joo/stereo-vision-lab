# Stereo Vision Lab

An interactive web application for exploring core stereo vision and epipolar geometry algorithms, built with FastAPI and OpenCV.

## Overview

Stereo Vision Lab lets you upload a pair of stereo images and walk through the classic computer vision pipeline — from dense disparity estimation to feature matching, fundamental matrix estimation, epipolar geometry, and uncalibrated rectification — all visualised directly in the browser.

## Features

| Module | Description |
| --- | --- |
| **Stereo Matching** | Dense disparity map via SSD block matching |
| **Feature Detection** | SIFT keypoint detection with ratio-test matching |
| **Fundamental Matrix** | Estimation via the normalised 8-point algorithm |
| **Epipolar Geometry** | Epipole computation and visualisation of epipolar lines |
| **Rectification** | Uncalibrated stereo rectification with re-computed disparity |

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Computer Vision:** OpenCV, NumPy
- **Frontend:** Jinja2 templates, HTML/CSS/JavaScript

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
git clone https://github.com/shoaib-joo/stereo-vision-lab
cd stereo_app
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and set a secret key:

```bash
cp .env.example .env
```

### Run

```bash
python app.py
```

The app will be available at [http://localhost:5050](http://localhost:5050).

## Deployment

This project includes a `Procfile` for deployment on platforms such as [Render](https://render.com):

1. Push the repository to GitHub.
2. Create a new Web Service on Render and connect the repository.
3. Configure the service:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app:app --workers 2 --timeout-keep-alive 120 --host 0.0.0.0 --port $PORT`
   - **Environment variable:** `SECRET_KEY` — a long, random string
4. Deploy.

## Project Structure

```
stereo_app/
├── app.py            # FastAPI application and routes
├── stereo_core.py    # Stereo vision and geometry algorithms
├── templates/         # Jinja2 templates (frontend UI)
├── static/            # Static assets and default sample images
├── session_data/      # Per-session working files (generated at runtime)
├── requirements.txt   # Python dependencies
└── Procfile           # Deployment process definition
```

## License

This project is for educational and demonstration purposes.
