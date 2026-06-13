import os, uuid, base64, time, shutil
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv

load_dotenv()  # loads .env locally; on Render, env vars are set in the dashboard
from stereo_core import (
    img_to_b64, stereo_matching, disparity_to_color,
    detect_and_match_features, draw_matches_image,
    fundamental_normalized_8point, compute_epipoles,
    epipolar_error, draw_epipolar_lines, rectify_images
)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'stereo-dev-key-change-in-prod')

BASE_DIR    = os.path.dirname(__file__)
SESSIONS_DIR = os.path.join(BASE_DIR, 'session_data')
DEFAULTS_DIR = os.path.join(BASE_DIR, 'static', 'defaults')
os.makedirs(SESSIONS_DIR, exist_ok=True)

# ── session helpers ──────────────────────────────────────────────────────────

def get_session_dir():
    sid = session.get('sid')
    if not sid:
        sid = str(uuid.uuid4())
        session['sid'] = sid
    d = os.path.join(SESSIONS_DIR, sid)
    os.makedirs(d, exist_ok=True)
    return d

def session_path(filename):
    return os.path.join(get_session_dir(), filename)

def save_array(name, arr):
    np.save(session_path(name + '.npy'), arr)

def load_array(name):
    p = session_path(name + '.npy')
    if not os.path.exists(p):
        raise FileNotFoundError(f'{name} not computed yet')
    return np.load(p)

def read_img(side):
    p = session_path(f'{side}.jpg')
    img = cv2.imread(p)
    if img is None:
        raise ValueError(f'Image {side} not found — load images first')
    return img

# ── routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/load_defaults', methods=['POST'])
def load_defaults():
    d = get_session_dir()
    shutil.copy(os.path.join(DEFAULTS_DIR, 'left.jpg'),  os.path.join(d, 'left.jpg'))
    shutil.copy(os.path.join(DEFAULTS_DIR, 'right.jpg'), os.path.join(d, 'right.jpg'))
    result = {}
    for side in ('left', 'right'):
        img = read_img(side)
        result[f'{side}_thumb'] = img_to_b64(img)
    return jsonify(result)


@app.route('/upload', methods=['POST'])
def upload():
    d = get_session_dir()
    for side in ('left', 'right'):
        f = request.files.get(side)
        if f:
            f.save(os.path.join(d, f'{side}.jpg'))
    result = {}
    for side in ('left', 'right'):
        img = read_img(side)
        result[f'{side}_thumb'] = img_to_b64(img)
    return jsonify(result)


@app.route('/run_stereo', methods=['POST'])
def run_stereo():
    data = request.json or {}
    win  = int(data.get('window_size', 15))
    maxd = int(data.get('max_disparity', 64))
    try:
        left_bgr  = read_img('left')
        right_bgr = read_img('right')
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    left_g  = cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2GRAY)
    right_g = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    t0   = time.time()
    disp = stereo_matching(left_g, right_g, win, maxd)
    elapsed = round(time.time() - t0, 2)

    colored = disparity_to_color(disp)
    return jsonify({
        'disparity_img': img_to_b64(colored),
        'elapsed':   elapsed,
        'min_disp':  float(disp.min()),
        'max_disp':  float(disp.max()),
        'mean_disp': round(float(disp.mean()), 2),
    })


@app.route('/run_features', methods=['POST'])
def run_features():
    try:
        left_bgr  = read_img('left')
        right_bgr = read_img('right')
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    left_g  = cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2GRAY)
    right_g = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    x, xp, kp1, kp2, good = detect_and_match_features(left_g, right_g)
    save_array('x',  x)
    save_array('xp', xp)

    # serialise keypoints & matches as arrays
    kp1_arr = np.array([[k.pt[0], k.pt[1]] for k in kp1])
    kp2_arr = np.array([[k.pt[0], k.pt[1]] for k in kp2])
    good_arr = np.array([[m.queryIdx, m.trainIdx, m.distance] for m in good])
    save_array('kp1',  kp1_arr)
    save_array('kp2',  kp2_arr)
    save_array('good', good_arr)

    match_img = draw_matches_image(left_bgr, right_bgr, kp1, kp2, good)
    return jsonify({'match_img': img_to_b64(match_img), 'n_matches': len(good)})


@app.route('/run_fundamental', methods=['POST'])
def run_fundamental():
    try:
        x  = load_array('x')
        xp = load_array('xp')
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 400

    F = fundamental_normalized_8point(x, xp)
    save_array('F', F)

    e_left, e_right = compute_epipoles(F)
    err = epipolar_error(F, x, xp)
    return jsonify({
        'F': F.tolist(),
        'epipole_left':  [round(float(e_left[0]),1),  round(float(e_left[1]),1)],
        'epipole_right': [round(float(e_right[0]),1), round(float(e_right[1]),1)],
        'mean_error': round(err, 6),
    })


@app.route('/run_epipolar', methods=['POST'])
def run_epipolar():
    try:
        F  = load_array('F')
        x  = load_array('x')
        xp = load_array('xp')
        left_bgr  = read_img('left')
        right_bgr = read_img('right')
    except (FileNotFoundError, ValueError) as e:
        return jsonify({'error': str(e)}), 400

    n = int((request.json or {}).get('n_lines', 12))
    im1, im2 = draw_epipolar_lines(left_bgr, right_bgr, x, xp, F, n_lines=n)
    return jsonify({'epi_left': img_to_b64(im1), 'epi_right': img_to_b64(im2)})


@app.route('/run_rectify', methods=['POST'])
def run_rectify():
    try:
        F  = load_array('F')
        x  = load_array('x')
        xp = load_array('xp')
        left_bgr  = read_img('left')
        right_bgr = read_img('right')
    except (FileNotFoundError, ValueError) as e:
        return jsonify({'error': str(e)}), 400

    left_rect, right_rect, side, _ = rectify_images(left_bgr, right_bgr, x, xp, F)
    if left_rect is None:
        return jsonify({'error': 'Rectification failed — try with more feature matches'}), 500

    lr_g = cv2.cvtColor(left_rect,  cv2.COLOR_BGR2GRAY)
    rr_g = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
    disp_rect  = stereo_matching(lr_g, rr_g, window_size=11, max_disparity=64)
    disp_color = disparity_to_color(disp_rect)

    return jsonify({
        'side_by_side':   img_to_b64(side),
        'left_rect':      img_to_b64(left_rect),
        'right_rect':     img_to_b64(right_rect),
        'disparity_rect': img_to_b64(disp_color),
        'mean_disp_rect': round(float(disp_rect.mean()), 2),
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=False)
