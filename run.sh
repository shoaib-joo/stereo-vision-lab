#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Starting Stereo Vision Lab at http://localhost:5050"
python3 app.py
