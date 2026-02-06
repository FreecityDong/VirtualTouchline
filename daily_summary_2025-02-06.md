# VirtualTouchline 今日工作总结（2025-02-06）

## 一、完成内容

### 1) 主界面（`1.py`）
- 左侧：球场背景图（`ground.jpg`/`ground.png`）左对齐显示，按窗口等比缩放。
- 右侧：阵型选择与说明面板。
  - 阵型选择改为**单选框**，每行两项。
  - “优缺点概览”固定在右下区域，不随选择区位置变化。
  - 中文字体自动匹配（`PingFang SC`/`Heiti` 等），保证中文显示。
- 阵型说明来自 `formations_11v11_overview.md`，支持子级要点显示（如“克制/被克制”）。
- 左侧球场上根据所选阵型绘制球员点位，并显示**中文位置标签**。
  - 例：`4-3-3` 显示“左后卫/中卫/右后卫、左中场/中场/右中场、左边锋/中锋/右边锋”。
  - 门将标签为“门将”。

### 2) 阵型资料整理
- 新增 `formations_11v11_overview.md`：
  - 常见 11 人制阵型
  - 进攻/防守特点
  - 克制关系

### 3) 标准场地尺寸文档
- 新增 `standard_pitch_dimensions.md`：
  - 国际比赛标准尺寸
  - 禁区/小禁区/中圈/点球点等关键尺寸

### 4) 手势识别模块（独立运行）
新增 `gesture_hand_demo.py`，功能：
- 左侧显示 `ground.jpg`，掌心位置映射到白色光标点（限制在图像范围内）。
- 右上角显示摄像头画面与手部骨架点。
- 右下角显示当前动作（抓取/移动/放开/待机）。
- 使用 **MediaPipe Tasks API**（解决 `mp.solutions` 兼容问题）。

配套脚本：
- 新增 `run_gesture_demo.sh`：自动创建 `Python 3.11` 虚拟环境 `.venv_gesture` 并安装依赖。
- 增加 `certifi` 依赖，尝试解决模型下载 SSL 问题。
- 支持 `HAND_MODEL_PATH` 环境变量指向本地模型文件。

### 5) 资源调整
- `ground.png` 转为 `ground.jpg`（降低体积，优先加载 JPG）。

---

## 二、当前运行方式

### 1) 主界面（球场 + 阵型）
```bash
python3 1.py
```

### 2) 手势识别 Demo
推荐一键启动：
```bash
./run_gesture_demo.sh
```

或手动激活 venv：
```bash
source .venv_gesture/bin/activate
python gesture_hand_demo.py
```

---

## 三、已知问题与处理建议

1. **模型下载 SSL 证书问题**
   - 已在代码中加入 `certifi` 兜底。
   - 若仍失败，可手动下载模型：
     - `hand_landmarker.task` 放在项目根目录
     - 或设置环境变量 `HAND_MODEL_PATH=/path/to/hand_landmarker.task`

2. **mediapipe 在 Python 3.12 兼容性差**
   - 已切换到 `Python 3.11` 的专用虚拟环境 `.venv_gesture`。

---

## 四、下一步建议（可选）
1. 将手势光标与主界面（`1.py`）联动，实现阵型/球员拖拽交互。
2. 增加阵型站位网格与球员编号/名字。
3. 添加手势动作稳定器（连续 N 帧阈值）以减少抖动误判。
4. 将 `gesture_hand_demo.py` 封装为可复用模块（类/回调接口）。
