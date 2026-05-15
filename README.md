# 食物卡路里识别系统 / Food Calorie Recognition

基于 YOLOv8 的食物卡路里自动识别 Web 应用——上传食物图片，AI 识别食物类别，自动计算卡路里及营养成分，并提供运动消耗换算参考。

## 项目概览

```
食物卡路里识别网页/          ← Web 应用（Flask + YOLOv8）
Fooddataset/                ← 训练/验证/测试数据集（19类，58K+ 图片）
train.py                    ← YOLOv8 模型训练脚本
best.pt                     ← 已训练模型（Git LFS）
model_result/               ← 模型评估结果（F1/PR曲线等）
```

## Web 应用功能

| 功能 | 说明 |
|------|------|
| 简易登录 | 输入用户名即可使用，自动记忆 |
| 图片上传 | 支持点击上传与拖拽上传，JPG/JPEG/PNG |
| YOLO 检测 | 筛选最高置信度食物目标，绘制检测框 |
| 营养计算 | 19 类食物内置营养数据库（kcal/蛋白质/脂肪/碳水） |
| 运动换算 | 自动换算步行、慢跑消耗时长 |
| 阈值调节 | 滑块调节置信度阈值（0.2–0.8） |
| 历史记录 | 浏览器本地存储，支持缩略图、删除、清空 |
| 分栏对比 | 原图 vs 检测结果图并列展示 |

### 启动方式

```bash
cd 食物卡路里识别网页
pip install flask ultralytics pillow
python app.py
```

浏览器自动打开 `http://127.0.0.1:5000`，输入任意用户名即可使用。

### UI 设计

- 配色：薰衣草紫色主题（`#8b5cf6`），渐变背景 + 毛玻璃导航栏
- 动画：登录卡片淡入、上传框呼吸闪烁、结果卡片滑入、历史列表错落入场
- 页面：登录页 → 主页（识别） → 预览页 → 历史记录页，独立路由

## 数据集

19 类中国常见食物，共计 58,621 张标注图片。

| 类别 | 类别 | 类别 |
|------|------|------|
| 面条主食 | 汉堡快餐 | 三明治 |
| 寿司卷物 | 鸡肉料理（含炸鸡） | 烧串烤肉 |
| 鸡蛋料理 | 熟制海鲜 | 刺身生食 |
| 汤锅炖物 | 蔬菜料理 | 豆腐料理 |
| 蛋糕甜品 | 面包点心 | 牛肉菜式 |
| 猪肉菜式 | 带馅早点 | 炸货点心 |
| 米饭主食 | | |

| 数据集 | 图片数 | 用途 |
|--------|--------|------|
| train | 54,048 | 模型训练 |
| valid_new | 2,287 | 模型验证 |
| test | 2,286 | 模型测试 |

## 模型训练

基于 YOLOv8n 预训练模型微调。

```python
model = YOLO("yolov8mm.pt")
model.train(
    data="Fooddataset/data.yaml",
    epochs=10,
    imgsz=480,
    batch=8,
    device="cpu",
    optimizer="AdamW",
    lr0=0.001,
)
```

训练脚本见 `train.py`，最佳权重保存为 `best.pt`（104 MB，Git LFS 管理）。

## 技术栈

| 层 | 技术 |
|----|------|
| 后端 | Python / Flask |
| AI 推理 | Ultralytics YOLOv8 |
| 前端 | HTML5 + CSS3 + JavaScript（无框架） |
| 存储 | localStorage（历史记录） |
| 版本管理 | Git + Git LFS |

## 仓库结构

```
Cyanine/
├── README.md
├── .gitignore
├── .gitattributes            ← Git LFS 配置
├── train.py                  ← 模型训练脚本
├── best.pt                   ← 训练好的 YOLOv8 模型（LFS）
├── Fooddataset/
│   ├── data.yaml             ← 数据集配置
│   ├── train/                ← 训练集（images + labels）
│   ├── valid_new/            ← 验证集
│   └── test/                 ← 测试集
├── model_result/
│   └── final_test_result_FINAL/
│       ├── F1_curve.png
│       ├── PR_curve.png
│       ├── confusion_matrix.png
│       ├── predictions.json
│       └── ...
└── 食物卡路里识别网页/
    ├── app.py                ← Flask 主程序
    └── templates/
        ├── login.html        ← 登录页
        ├── index.html        ← 主页（上传+识别+结果）
        ├── preview.html      ← 图片预览页
        └── history.html      ← 历史记录页
```

## 许可证

数据集标注采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可。
