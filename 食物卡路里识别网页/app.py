"""
食物卡路里识别系统 — Flask 后端
"""
import os
import uuid
import glob as gb
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

MODEL_PATH = r"D:\Cyanine\best.pt"
model = YOLO(MODEL_PATH)

TEMP_DIR = os.path.join(app.static_folder or "static", "upload")
os.makedirs(TEMP_DIR, exist_ok=True)

# 启动时清理旧临时文件
for old in gb.glob(os.path.join(TEMP_DIR, "*")):
    try:
        os.remove(old)
    except Exception:
        pass

CLASS_NAMES = [
    "面条主食", "汉堡快餐", "三明治", "寿司卷物",
    "鸡肉料理（含炸鸡）", "烧串烤肉", "鸡蛋料理", "熟制海鲜",
    "刺身生食", "汤锅炖物", "蔬菜料理", "豆腐料理",
    "蛋糕甜品", "面包点心", "牛肉菜式", "猪肉菜式",
    "带馅早点", "炸货点心", "米饭主食"
]

NUTRITION_DB = {
    "面条主食":   {"kcal": 110, "protein": 4.0, "fat": 0.8, "carbs": 22.0, "portion": 250, "portion_unit": "g/碗"},
    "汉堡快餐":   {"kcal": 295, "protein": 14.0, "fat": 14.0, "carbs": 28.0, "portion": 200, "portion_unit": "g/个"},
    "三明治":    {"kcal": 230, "protein": 10.0, "fat": 9.0, "carbs": 26.0, "portion": 150, "portion_unit": "g/个"},
    "寿司卷物":   {"kcal": 145, "protein": 6.0, "fat": 2.0, "carbs": 26.0, "portion": 200, "portion_unit": "g/份"},
    "鸡肉料理（含炸鸡）": {"kcal": 200, "protein": 20.0, "fat": 12.0, "carbs": 3.0, "portion": 200, "portion_unit": "g/份"},
    "烧串烤肉":   {"kcal": 220, "protein": 18.0, "fat": 16.0, "carbs": 2.0, "portion": 150, "portion_unit": "g/份"},
    "鸡蛋料理":   {"kcal": 155, "protein": 12.0, "fat": 10.0, "carbs": 2.0, "portion": 150, "portion_unit": "g/份"},
    "熟制海鲜":   {"kcal": 110, "protein": 18.0, "fat": 3.0, "carbs": 2.0, "portion": 200, "portion_unit": "g/份"},
    "刺身生食":   {"kcal": 100, "protein": 20.0, "fat": 1.5, "carbs": 0.5, "portion": 150, "portion_unit": "g/份"},
    "汤锅炖物":   {"kcal": 90,  "protein": 8.0, "fat": 4.0, "carbs": 6.0, "portion": 400, "portion_unit": "g/锅"},
    "蔬菜料理":   {"kcal": 65,  "protein": 3.0, "fat": 3.0, "carbs": 7.0, "portion": 200, "portion_unit": "g/份"},
    "豆腐料理":   {"kcal": 80,  "protein": 8.0, "fat": 4.0, "carbs": 3.0, "portion": 200, "portion_unit": "g/份"},
    "蛋糕甜品":   {"kcal": 370, "protein": 5.0, "fat": 18.0, "carbs": 48.0, "portion": 100, "portion_unit": "g/块"},
    "面包点心":   {"kcal": 310, "protein": 8.0, "fat": 10.0, "carbs": 46.0, "portion": 80,  "portion_unit": "g/个"},
    "牛肉菜式":   {"kcal": 190, "protein": 22.0, "fat": 10.0, "carbs": 2.0, "portion": 200, "portion_unit": "g/份"},
    "猪肉菜式":   {"kcal": 240, "protein": 18.0, "fat": 18.0, "carbs": 1.0, "portion": 200, "portion_unit": "g/份"},
    "带馅早点":   {"kcal": 220, "protein": 8.0, "fat": 10.0, "carbs": 26.0, "portion": 200, "portion_unit": "g/份"},
    "炸货点心":   {"kcal": 400, "protein": 6.0, "fat": 24.0, "carbs": 40.0, "portion": 100, "portion_unit": "g/份"},
    "米饭主食":   {"kcal": 116, "protein": 2.6, "fat": 0.3, "carbs": 25.6, "portion": 200, "portion_unit": "g/碗"},
}


@app.route("/")
def login():
    return render_template("login.html")


@app.route("/main")
def main_page():
    return render_template("index.html")


@app.route("/preview")
def preview_page():
    return render_template("preview.html")


@app.route("/history")
def history_page():
    return render_template("history.html")


@app.route("/upload", methods=["POST"])
def upload():
    """上传图片，暂存到临时目录，返回 temp_id"""
    if "image" not in request.files:
        return jsonify({"error": "未收到图片"}), 400

    file = request.files["image"]
    ext = os.path.splitext(file.filename or "img.jpg")[1] or ".jpg"
    temp_id = uuid.uuid4().hex + ext
    filepath = os.path.join(TEMP_DIR, temp_id)

    img = Image.open(file.stream).convert("RGB")
    img.save(filepath)

    return jsonify({"temp_id": temp_id, "ok": True})


@app.route("/temp/<temp_id>")
def serve_temp(temp_id):
    """提供临时图片文件"""
    return send_from_directory(TEMP_DIR, temp_id)


@app.route("/predict", methods=["POST"])
def predict():
    """接收 temp_id，执行 YOLO 推理"""
    temp_id = request.form.get("temp_id", "")
    if not temp_id:
        return jsonify({"error": "缺少 temp_id"}), 400

    filepath = os.path.join(TEMP_DIR, temp_id)
    if not os.path.exists(filepath):
        return jsonify({"error": "图片已过期，请重新上传"}), 404

    conf = float(request.form.get("conf", 0.3))
    img = Image.open(filepath).convert("RGB")

    results = model(img, conf=conf, verbose=False)

    detections = []
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf_val = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            detections.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "未知",
                "confidence": round(conf_val, 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            })

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    top_detection = None
    if detections:
        top = detections[0]
        nutrition = NUTRITION_DB.get(top["class_name"])
        top_detection = {
            "class_name": top["class_name"],
            "confidence": top["confidence"],
            "bbox": top["bbox"],
            "nutrition": nutrition
        }

    return jsonify({
        "detections": detections,
        "top_detection": top_detection,
        "total_count": len(detections)
    })


if __name__ == "__main__":
    import webbrowser
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)
