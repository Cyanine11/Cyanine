from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(
    data="/root/autodl-tmp/Fooddataset/data.yaml",

    epochs=120,
    batch=24,
    imgsz=640,

    device="cuda",
    amp=True,

    workers=8,
    cache="disk",

    patience=25,
    save_period=10,
    seed=42,

    optimizer="SGD",

    lr0=0.004,
    lrf=0.05,
    cos_lr=True,

    warmup_epochs=3,

    weight_decay=0.0005,

    hsv_h=0.015,
    hsv_s=0.35,
    hsv_v=0.25,

    fliplr=0.5,

    scale=0.5,
    translate=0.1,

    mosaic=1.0,
    mixup=0.03,
    copy_paste=0.0,

    close_mosaic=15,

    box=7.5,
    cls=0.5,
    dfl=1.5,

    verbose=True,
    plots=True,

    project="food_detect_safe",
    name="yolov8m_final_v5",
    exist_ok=True
)