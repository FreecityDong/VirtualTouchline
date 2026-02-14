# 项目目录结构

## 顶层目录

- `src/virtualtouchline/`：正式程序源码（主应用、手势交互、状态机）
- `images/`：运行素材（球场图、球员头像、足球图标）
- `models/`：模型文件（`hand_landmarker.task`）
- `docs/`：说明文档、设计文档、数据说明、手册
- `history/`：历史/实验脚本（不参与正式运行）
- `run_gesture_demo.sh`：一键运行脚本（默认启动融合主程序）
- `requirements.txt`：依赖列表

## 源码模块（`src/virtualtouchline`）

- `fusion_pygame_simulator.py`：融合主程序入口（推荐运行）
- `gesture_formation_interaction.py`：手势与球员交互基础能力
- `match_state_machine_simulator.py`：比赛状态机与分析上下文
- `paths.py`：统一资源路径配置（素材/模型/文档）
- `__main__.py`：支持 `python -m src.virtualtouchline`

## 兼容入口（保留）

为兼容原命令，根目录保留了薄包装入口：

- `fusion_pygame_simulator.py`
- `gesture_formation_interaction.py`
- `match_state_machine_simulator.py`

因此以下命令依然可用：

```bash
python3 fusion_pygame_simulator.py
```

