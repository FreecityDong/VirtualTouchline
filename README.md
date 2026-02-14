# VirtualTouchline

基于手势交互的足球战术模拟器。  
主程序为 `fusion_pygame_simulator.py`，支持比赛推进、阵型/策略调整、时间窗配额限制，以及中场/终场报告弹窗。

## 主要功能

- 手势拖拽 A 队球员并用 `OK` 手势确认阵型
- A 队策略分组互斥选择（抓取 0.5 秒选中，`OK` 1 秒确认提交）
- 时间窗配额控制（每 15 分钟比赛时间窗）
- 自动推进/手动推进/快速快进（`M`）
- 中场（45:00）与终场（90:00）报告自动生成与弹窗展示
- 报告弹窗可 `OK` 1 秒关闭，或自动关闭（30 秒）

## 目录结构

```text
VirtualTouchline/
├─ src/virtualtouchline/          # 正式源码
│  ├─ fusion_pygame_simulator.py
│  ├─ gesture_formation_interaction.py
│  ├─ match_state_machine_simulator.py
│  └─ paths.py
├─ images/                        # 素材
│  ├─ ground.jpg
│  └─ players/
├─ models/
│  └─ hand_landmarker.task
├─ docs/                          # 文档
├─ history/                       # 历史/实验脚本
├─ fusion_pygame_simulator.py     # 根目录兼容入口
├─ run_gesture_demo.sh
└─ requirements.txt
```

## 环境安装

```bash
python3 -m venv .venv_gesture
source .venv_gesture/bin/activate
pip install -r requirements.txt
```

## 运行方式

推荐（兼容入口）：

```bash
python3 fusion_pygame_simulator.py
```

或：

```bash
bash run_gesture_demo.sh
```

## 键盘快捷键

- `Space`：开始/暂停自动推进
- `N`：手动推进 15 秒比赛时间
- `M`：快进到中场/终场
- `R`：重置比赛

## 手势交互规则

- 底部按钮：光标悬停 + 抓取 1 秒触发
- 策略项：光标悬停 + 抓取 0.5 秒选中
- 策略提交：`OK` 手势保持 1 秒确认
- 报告弹窗：`OK` 手势保持 1 秒关闭

## 报告说明

- 自动触发：
  - 45:00 中场报告
  - 90:00 终场报告
- 弹窗期间比赛会暂停
- 已生成报告可通过底部按钮再次打开
- 若大模型不可用，会自动回退到本地摘要报告

## 依赖与接口

- 核心依赖：`pygame`、`mediapipe`、`opencv-python`、`pillow`
- 报告生成：`openai` + DashScope 兼容接口
- API Key：
  - 优先读取环境变量 `DASHSCOPE_API_KEY`
  - 未设置时使用程序内默认值

## 常见问题

- 摄像头无画面：检查系统摄像头权限与设备占用
- 手势难命中按钮：确保手部在画面中、抓取手势稳定保持
- 报告未生成：先确认比赛时间是否到达 45:00/90:00

