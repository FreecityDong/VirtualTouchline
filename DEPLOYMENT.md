# 部署说明（本地运行 / 演示环境）

本文档用于在本机或演示机器上快速部署并运行项目的两个核心入口：
- `1.py`（主界面：球场 + 阵型 + 中文标签）
- `gesture_hand_demo.py`（手势识别 Demo）

---

## 一、系统要求

- macOS / Windows / Linux
- Python 3.10 或 3.11（**推荐 3.11**，MediaPipe 兼容性更好）
- 摄像头（用于手势识别模块）

---

## 二、推荐部署方式（使用内置脚本）

### 1) 手势识别 Demo（自动创建 venv）
```bash
./run_gesture_demo.sh
```
该脚本会：
- 创建 `.venv_gesture`
- 安装 `mediapipe / opencv-python / pillow / certifi`
- 运行 `gesture_hand_demo.py`

> 若第一次运行下载模型失败，见“模型下载问题”章节。

### 2) 主界面运行
```bash
python3 1.py
```

---

## 三、手动部署方式（可选）

### 1) 创建虚拟环境
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2) 安装依赖
```bash
python -m pip install -r requirements.txt
```

### 3) 运行
```bash
python 1.py
python gesture_hand_demo.py
```

---

## 四、模型下载问题（SSL / 证书）

`gesture_hand_demo.py` 首次运行会下载 `hand_landmarker.task`：

下载地址：
```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

若遇到 SSL 错误：
1. 确认已安装 `certifi`  
   ```bash
   python -m pip install certifi
   ```
2. 手动下载模型并放到项目根目录：
   ```bash
   curl -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o hand_landmarker.task
   ```
3. 或使用环境变量指定模型路径：
   ```bash
   export HAND_MODEL_PATH=/path/to/hand_landmarker.task
   ```

---

## 五、摄像头权限（macOS）

若摄像头无法打开：
- 前往「系统设置 → 隐私与安全性 → 相机」
- 允许当前终端/IDE 访问摄像头

---

## 六、资源文件

确保以下文件存在：
- `ground.jpg`（球场背景，优先使用）
- `ground.png`（备选）
- `formations_11v11_overview.md`（阵型说明）

---

## 七、常见问题

### 1) `mediapipe` 导入失败
建议使用 Python 3.11 并确保安装源正确：
```bash
python -m pip uninstall mediapipe -y
python -m pip install mediapipe
```

### 2) 中文不显示
若系统缺少中文字体，可安装 `Noto Sans CJK`，或在代码中指定字体路径。

---

如需进一步部署到演示大屏或融合主程序与手势模块，请告诉我你的目标环境（设备、分辨率、系统），我可以补充更细化的部署指导。
