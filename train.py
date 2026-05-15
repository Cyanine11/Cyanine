import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="UECFOOD256.v1-japon.yolov8/data.yaml",
    epochs=10,
    imgsz=480,
    batch=8,           # 这里改成8，比2快4倍，比16更稳
    device="cpu",
    workers=4,         # 改成4，和batch匹配，避免数据加载瓶颈
    amp=False,         # CPU上不开启混合精度，避免不稳定
    optimizer="AdamW", # 换更稳定的优化器，弥补batch稍小的不足
    lr0=0.001,         # 调低学习率，配合小batch，训练更稳
    patience=5,
    project="food_train",
    name="exp_optimal"
)