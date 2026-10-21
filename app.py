from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
import uuid
import json
import math
import hashlib
import json
import re
import secrets
import time
import base64
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)

app.secret_key = "palm_payment_secret_key_2026"


# =========================================================
# SEPARATE CUSTOMER / MERCHANT PORTAL MODES
# =========================================================

AUTH_SAFE_ENDPOINTS = {
    "home",
    "login",
    "register",
    "merchant_login",
    "merchant_register",
    "logout",
    "verify_receipt",
    "receipt_api",
    "static",
}


@app.before_request
def enforce_portal_mode():
    mode = session.get("portal_mode")
    if not mode:
        return None

    endpoint = request.endpoint or ""

    if endpoint in AUTH_SAFE_ENDPOINTS:
        return None

    if mode == "merchant":
        if (
            endpoint == "merchant_dashboard"
            or request.path.startswith("/api/merchant")
            or endpoint == "security_live_monitor"
        ):
            return None
        return redirect(url_for("merchant_dashboard"))

    if mode == "customer":
        if endpoint == "merchant_dashboard" or request.path.startswith("/api/merchant"):
            return redirect(url_for("dashboard"))
        return None

    if mode == "admin":
        if endpoint == "admin_dashboard" or request.path.startswith("/api/admin"):
            return None
        return redirect(url_for("admin_dashboard"))

    return None


# =========================================================
# CCNET CONTACTLESS PALM RECOGNITION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
CCNET_MODEL_PATH = PROJECT_ROOT / "models" / "palm_ccnet.onnx"
# Demo threshold only. Calibrate this against your own webcam captures.
CCNET_SIMILARITY_THRESHOLD = float(os.getenv("PALM_CCNET_THRESHOLD", "0.60"))
_ccnet_session = None


def get_ccnet_session():
    """Load the CCNet ONNX model lazily so the Flask app can still start."""
    global _ccnet_session

    if _ccnet_session is not None:
        return _ccnet_session

    if ort is None:
        raise RuntimeError("onnxruntime is not installed. Run: pip install onnxruntime")

    if cv2 is None or np is None:
        raise RuntimeError("opencv-python and numpy are required for CCNet inference.")

    if not CCNET_MODEL_PATH.exists():
        raise RuntimeError(f"CCNet model not found: {CCNET_MODEL_PATH}")

    _ccnet_session = ort.InferenceSession(
        str(CCNET_MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )
    return _ccnet_session


def _decode_image_payload(image_payload):
    """Decode a browser data URL/base64 image into an OpenCV image."""
    if not isinstance(image_payload, str) or not image_payload.strip():
        raise ValueError("Palm image is missing.")

    raw = image_payload.strip()
    if raw.startswith("data:"):
        if "," not in raw:
            raise ValueError("Invalid palm image data URL.")
        raw = raw.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 palm image.") from exc

    if not raw_bytes:
        raise ValueError("Palm image is empty.")

    array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode palm image.")

    return image


def ccnet_embedding_from_image(image_payload):
    """Convert one palm crop into the 2048-D CCNet embedding."""
    image = _decode_image_payload(image_payload)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    tensor = gray.astype(np.float32) / 255.0
    tensor = tensor[None, None, :, :]  # [1, 1, 128, 128]

    session_ = get_ccnet_session()
    input_name = session_.get_inputs()[0].name
    output = session_.run(None, {input_name: tensor})[0]
    embedding = np.asarray(output, dtype=np.float32).reshape(-1)

    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-12:
        raise ValueError("CCNet returned an invalid zero embedding.")

    return (embedding / norm).tolist()


def average_ccnet_embeddings(embeddings):
    if not embeddings:
        return None

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 2048:
        raise ValueError("Invalid CCNet embedding dimensions.")

    average = matrix.mean(axis=0)
    norm = float(np.linalg.norm(average))
    if norm <= 1e-12:
        raise ValueError("Unable to create a stable CCNet registration template.")

    return (average / norm).tolist()


def cosine_similarity(left, right):
    left_np = np.asarray(left, dtype=np.float32)
    right_np = np.asarray(right, dtype=np.float32)

    if left_np.size != 2048 or right_np.size != 2048:
        raise ValueError("Invalid CCNet embedding length.")

    left_norm = np.linalg.norm(left_np)
    right_norm = np.linalg.norm(right_np)
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0

    return float(np.dot(left_np, right_np) / (left_norm * right_norm))


# =========================================================
# CCNet PALMPRINT MODEL
# =========================================================
CCNET_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "palm_ccnet.onnx")
ccnet_session = None
ccnet_input_name = None
ccnet_output_name = None


def get_ccnet_session():
    """Load the CCNet ONNX palmprint feature extractor once."""
    global ccnet_session, ccnet_input_name, ccnet_output_name

    if ccnet_session is not None:
        return ccnet_session

    if not os.path.exists(CCNET_MODEL_PATH):
        raise FileNotFoundError(f"CCNet model not found: {CCNET_MODEL_PATH}")

    ccnet_session = ort.InferenceSession(
        CCNET_MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )
    ccnet_input_name = ccnet_session.get_inputs()[0].name
    ccnet_output_name = ccnet_session.get_outputs()[0].name
    return ccnet_session


def decode_palm_image(image_data):
    """Decode a browser data URL/base64 image into an OpenCV BGR image."""
    if not image_data or not isinstance(image_data, str):
        raise ValueError("Palm image was not provided.")

    encoded = image_data.split(",", 1)[1] if "," in image_data else image_data
    raw = base64.b64decode(encoded, validate=True)
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Palm image could not be decoded.")
    return image


def normalize_landmarks(raw_landmarks):
    """Validate 21 MediaPipe landmarks supplied by the browser."""
    if not isinstance(raw_landmarks, list) or len(raw_landmarks) != 63:
        raise ValueError("Expected 21 palm landmarks.")

    values = [float(v) for v in raw_landmarks]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Palm landmarks contain invalid numbers.")
    return values


def crop_palm_roi(image, landmarks):
    """Create a stable palm ROI from wrist/MCP landmarks.

    The browser's MediaPipe landmarks are normalized to [0,1]. We use the
    wrist and four MCP joints to avoid letting the fingertips dominate the
    palmprint crop. This is a project prototype; a production system should
    use a dedicated palm ROI/alignment model.
    """
    h, w = image.shape[:2]
    points = np.asarray(landmarks, dtype=np.float32).reshape(21, 3)

    # Wrist + index/middle/ring/little MCP joints.
    palm_indices = [0, 5, 9, 13, 17]
    xs = np.clip(points[palm_indices, 0], 0.0, 1.0) * w
    ys = np.clip(points[palm_indices, 1], 0.0, 1.0) * h

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    bw = max(1.0, x_max - x_min)
    bh = max(1.0, y_max - y_min)

    # Expand enough to include the full palm surface.
    pad_x = max(0.12 * bw, 0.05 * w)
    pad_y = max(0.18 * bh, 0.05 * h)

    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(w, int(x_max + pad_x))
    y2 = min(h, int(y_max + pad_y))

    roi = image[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
        raise ValueError("Palm ROI is too small. Keep your palm inside the camera frame.")
    return roi


def palm_embedding_from_image(image_data, landmarks):
    """Generate the 2048-D L2-normalized CCNet palm embedding."""
    image = decode_palm_image(image_data)
    values = normalize_landmarks(landmarks)
    roi = crop_palm_roi(image, values)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    tensor = gray.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, np.newaxis, :, :]

    session_model = get_ccnet_session()
    embedding = session_model.run(
        [ccnet_output_name],
        {ccnet_input_name: tensor}
    )[0]
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)

    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-12 or not np.isfinite(norm):
        raise ValueError("CCNet returned an invalid palm embedding.")

    embedding = embedding / norm
    if embedding.size != 2048:
        raise ValueError(f"Unexpected CCNet embedding size: {embedding.size}")
    return embedding.tolist()


def cosine_similarity(a, b):
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    if va.size != 2048 or vb.size != 2048:
        raise ValueError("Invalid CCNet embedding size.")
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 1e-12 or nb <= 1e-12:
        return -1.0
    return float(np.dot(va, vb) / (na * nb))


def average_embeddings(embeddings):
    """Average multiple CCNet embeddings and L2-normalize the template."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 2048 or matrix.shape[0] < 1:
        raise ValueError("Invalid CCNet embedding collection.")
    mean = matrix.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= 1e-12 or not np.isfinite(norm):
        raise ValueError("Could not create a valid palm template.")
    return (mean / norm).tolist()


# =========================================================
# DATABASE
# =========================================================

DB_PATH = PROJECT_ROOT / "palm_payment.db"


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            mobile TEXT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            palm_id TEXT,
            palm_template TEXT,
            account_status TEXT NOT NULL DEFAULT 'ACTIVE'
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            receiver TEXT NOT NULL,
            receiver_name TEXT,
            amount REAL NOT NULL,
            status TEXT NOT NULL,
            transaction_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS risk_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            receiver TEXT NOT NULL,
            amount REAL NOT NULL,
            risk_score INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            reasons TEXT,
            action TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            amount REAL,
            transaction_id TEXT,
            is_read INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS security_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            activity TEXT NOT NULL,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS virtual_cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL UNIQUE,
            card_number TEXT NOT NULL UNIQUE,
            expiry TEXT NOT NULL,
            cvv TEXT NOT NULL,
            is_frozen INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS merchant_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL UNIQUE,
            merchant_id TEXT NOT NULL UNIQUE,
            business_name TEXT NOT NULL,
            category TEXT NOT NULL,
            accepting_payments INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # Advanced merchant features: product catalog, POS sales and refunds.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS merchant_products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            merchant_user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            sku TEXT,
            category TEXT NOT NULL DEFAULT 'Other',
            price REAL NOT NULL DEFAULT 0,
            stock INTEGER NOT NULL DEFAULT 0,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(merchant_user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS merchant_pos_sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            merchant_user_id INTEGER NOT NULL,
            order_id TEXT NOT NULL UNIQUE,
            customer_mobile TEXT,
            customer_name TEXT,
            amount REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'COMPLETED',
            transaction_id TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(merchant_user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS merchant_pos_sale_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pos_sale_id INTEGER NOT NULL,
            product_id INTEGER,
            product_name TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            line_total REAL NOT NULL,
            FOREIGN KEY(pos_sale_id) REFERENCES merchant_pos_sales(id),
            FOREIGN KEY(product_id) REFERENCES merchant_products(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS merchant_refunds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            merchant_user_id INTEGER NOT NULL,
            transaction_id TEXT NOT NULL,
            refund_id TEXT NOT NULL UNIQUE,
            original_amount REAL NOT NULL,
            refund_amount REAL NOT NULL,
            reason TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'COMPLETED',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(merchant_user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS palm_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            profile_name TEXT NOT NULL,
            relationship TEXT NOT NULL DEFAULT 'Family',
            palm_id TEXT NOT NULL UNIQUE,
            palm_template TEXT NOT NULL,
            permissions TEXT NOT NULL DEFAULT '{"payment":true}',
            active INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS palm_security_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            score REAL NOT NULL DEFAULT 0,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS admin_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'Administrator',
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS admin_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_ref TEXT NOT NULL,
            action TEXT NOT NULL,
            target TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    def add_column_if_missing(table_name, column_name, column_definition):
        existing_columns = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in existing_columns:
            conn.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
            )

    add_column_if_missing("users", "security_version", "INTEGER NOT NULL DEFAULT 1")
    add_column_if_missing("users", "payment_pin", "TEXT")
    add_column_if_missing("users", "palm_template", "TEXT")
    add_column_if_missing("users", "mobile", "TEXT")
    add_column_if_missing("users", "account_status", "TEXT NOT NULL DEFAULT 'ACTIVE'")
    add_column_if_missing(
        "merchant_profiles",
        "accepting_payments",
        "INTEGER NOT NULL DEFAULT 1"
    )
    add_column_if_missing(
        "merchant_profiles",
        "created_at",
        "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    )
    add_column_if_missing(
        "merchant_profiles",
        "updated_at",
        "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    )
    add_column_if_missing("transactions", "receiver_name", "TEXT")

    conn.commit()
    conn.close()


init_db()


# =========================================================
# INDIAN MOBILE NUMBER HELPERS
# =========================================================

def normalize_indian_mobile(value):
    if value is None:
        return None

    digits = re.sub(r"\D", "", str(value).strip())
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    if re.fullmatch(r"[6-9]\d{9}", digits):
        return digits
    return None


def format_indian_mobile(value):
    mobile = normalize_indian_mobile(value)
    if not mobile:
        return value or ""
    return "+91 " + mobile


@app.before_request
def enforce_session_security():
    if "user_id" not in session:
        return None

    public_endpoints = {"home", "login", "register", "register_palm", "save_palm", "static"}
    if request.endpoint in public_endpoints:
        return None

    conn = get_db()
    user = conn.execute("SELECT security_version FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    conn.close()

    if not user:
        session.clear()
        return redirect(url_for("login"))

    db_version = int(user["security_version"] or 1)
    session_version = session.get("security_version")

    if session_version is None:
        session["security_version"] = db_version
        return None

    if int(session_version) != db_version:
        session.clear()
        return redirect(url_for("login"))

    return None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        mobile = normalize_indian_mobile(request.form.get("mobile", ""))
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not mobile or not email or not password:
            return render_template("register.html", error="Please enter your name, valid Indian mobile number, email and password.")

        if len(password) < 6:
            return render_template("register.html", error="Password must contain at least 6 characters.")

        hashed_password = generate_password_hash(password)
        conn = get_db()
        try:
            cursor = conn.execute("""
                INSERT INTO users
                (name, mobile, email, password)
                VALUES (?, ?, ?, ?)
            """, (name, mobile, email, hashed_password))
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            session.clear()
            session["portal_mode"] = "customer"
            session["user_id"] = user_id
            session["security_version"] = 1
            session["user_name"] = name
            return redirect(url_for("register_palm"))
        except sqlite3.IntegrityError:
            conn.close()
            return render_template("register.html", error="Email already registered!")

    return render_template("register.html")


@app.route("/register-palm")
def register_palm():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("register_palm.html", name=session.get("user_name"))


@app.route("/save-palm", methods=["POST"])
def save_palm():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Please login first."}), 401

    data = request.get_json(silent=True) or {}
    legacy_template = data.get("template")
    handedness = data.get("handedness", "Unknown")
    palm_images = data.get("palm_images") or []

    if legacy_template is not None:
        if not isinstance(legacy_template, list) or len(legacy_template) != 63:
            return jsonify({"success": False, "message": "Invalid landmark template."}), 400

    if not isinstance(palm_images, list):
        return jsonify({"success": False, "message": "Invalid palm image samples."}), 400

    palm_images = [item for item in palm_images if isinstance(item, str) and item.strip()]
    if len(palm_images) > 10:
        palm_images = palm_images[:10]

    ccnet_embeddings = []

    if palm_images:
        try:
            for image_payload in palm_images:
                ccnet_embeddings.append(ccnet_embedding_from_image(image_payload))
            ccnet_template = average_ccnet_embeddings(ccnet_embeddings)
        except Exception as exc:
            app.logger.exception("CCNet registration failed")
            return jsonify({"success": False, "message": f"CCNet palm registration failed: {exc}"}), 500
    else:
        ccnet_template = None

    if ccnet_template is None and legacy_template is None:
        return jsonify({"success": False, "message": "No palm biometric samples received."}), 400

    palm_id = "PALM-" + uuid.uuid4().hex[:10].upper()
    stored_payload = {
        "version": 2,
        "method": "CCNet_2048D" if ccnet_template is not None else "MediaPipe_63D_LEGACY",
        "template": legacy_template,
        "ccnet_embedding": ccnet_template,
        "ccnet_samples": len(ccnet_embeddings),
        "handedness": handedness
    }

    conn = get_db()
    conn.execute("""
        UPDATE users
        SET palm_id = ?,
            palm_template = ?
        WHERE id = ?
    """, (palm_id, json.dumps(stored_payload), session["user_id"]))

    conn.execute("UPDATE palm_profiles SET active = 0, updated_at = CURRENT_TIMESTAMP WHERE user_id = ? AND relationship = 'Primary'", (session["user_id"],))
    conn.execute("""
        INSERT INTO palm_profiles
        (user_id, profile_name, relationship, palm_id, palm_template, permissions, active)
        VALUES (?, ?, 'Primary', ?, ?, ?, 1)
    """, (session["user_id"], "Primary Palm", palm_id, json.dumps(stored_payload), json.dumps({"payment": True})))

    conn.commit()
    conn.close()

    session["palm_id"] = palm_id
    log_security_activity(session["user_id"], "Palm registered", f"Palm biometric template created using {'CCNet' if ccnet_template is not None else 'legacy landmark'} mode")

    return jsonify({"success": True, "message": "Palm registered successfully.", "palm_id": palm_id, "biometric_mode": "CCNet_2048D" if ccnet_template is not None else "MediaPipe_63D_LEGACY", "samples": len(ccnet_embeddings) if ccnet_embeddings else 0})


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()
        if user and check_password_hash(user["password"], password):
            session["portal_mode"] = "customer"
            session["user_id"] = user["id"]
            session["security_version"] = int(user["security_version"] or 1)
            session["user_name"] = user["name"]
            session["user_mobile"] = user["mobile"]
            session["palm_id"] = user["palm_id"]
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid email or password!")
    return render_template("login.html")


# This repo snapshot is intentionally kept functionally aligned with the original design and is optimized for lean Git storage.

if __name__ == "__main__":
    init_db()
    app.run(debug=True)

