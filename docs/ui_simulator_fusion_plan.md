# UI 与模拟器融合方案（V1）

## 1. 背景与目标

当前项目已有两条成熟能力线：

- `gesture_formation_interaction.py`：手势识别 + 球场可视化 + 球员拖拽与阵型动画（Pygame）
- `match_state_machine_simulator.py`：状态机推演 + 战术控制台 + 统计看板 + 中场/终场AI分析（Tkinter）

本方案目标是在不破坏现有稳定性的前提下，完成“交互层 + 推演层 + 分析层”的统一产品体验：

1. 单一主程序入口
2. 手势操作直接驱动战术调整
3. 战术调整即时反映到状态机与看板
4. 中场/终场分析保持自动触发与弹窗展示

---

## 2. 设计约束与现状判断

## 2.1 技术现状

- 手势UI主循环：`pygame + cv2 + mediapipe`
- 模拟器主循环：`tkinter`
- 两端都具备完整主循环，需统一到单主循环框架

## 2.2 关键约束

- 主界面必须采用 `pygame`，`tkinter` 仅保留引擎逻辑可复用部分
- 融合优先级应为“状态统一 > 交互统一 > 视觉统一”
- 需要保持 `datamodel_state_machine_v2.md` 的规则一致性

## 2.3 融合原则

- **单一真相源（Single Source of Truth）**：状态机引擎 `MatchSimulatorEngine` 作为全局状态源
- **事件驱动（Event-Driven）**：手势侧只发“战术意图事件”，不直接改比赛状态
- **分层解耦（Layered）**：手势识别、战术应用、比赛推进、AI分析分层隔离
- **渐进式上线（Incremental）**：先打通功能闭环，再做视觉整合

---

## 3. 融合目标架构

## 3.1 总体结构

- **App Shell（Pygame）**
  - 以 `gesture_formation_interaction.py` 作为主窗口与主循环
  - 承载球场渲染、战术控制、统计看板、日志、AI弹窗
- **Gesture Adapter（后台线程）**
  - 从 `gesture_formation_interaction.py` 抽取“手势识别 + 编队识别 + 球员操作意图”
  - 输出标准化事件给主循环，不直接改引擎状态
- **Domain Engine（MatchSimulatorEngine）**
  - 统一处理阵型、策略、事件、统计、回流
- **Analysis Service（DeepSeek）**
  - 45/90分钟触发报告，异步生成并回传UI层

## 3.2 数据流

1. 摄像头输入 -> Gesture Adapter
2. Gesture Adapter 产生 `GestureCommand`
3. Pygame App Shell 接收命令并映射成 `TacticalCommand`
4. 调用 `engine.apply_team_adjustment(...)`
5. 看板、日志、统计、区域图实时更新
6. 中场/终场自动触发AI分析

---

## 4. 分层模块设计

## 4.1 交互层（Input Layer）

### 输入来源
- 手势：抓取/移动/放开/确认（OK）
- 控件：阵型单选、策略单选、推进按钮

### 输出约束
- 统一输出 `GestureCommand`，字段建议：
  - `timestamp`
  - `team`（A/B）
  - `command_type`（select_player/move_player/release/confirm/formation_detected）
  - `payload`

---

## 4.2 编排层（Orchestration Layer）

新增 `FusionController`（建议单文件或内嵌在 simulator）：

- 职责1：命令队列管理（去抖/节流/去重）
- 职责2：手势命令 -> 战术命令映射
- 职责3：时间窗规则前置校验（阵型1次、策略2次）
- 职责4：生成用户可读日志（命令来源、是否生效、原因）

---

## 4.3 领域层（Domain Layer）

复用 `MatchSimulatorEngine` 作为唯一状态源：

- 阵型特征：8维向量
- 策略组：高度/节奏/主攻通道
- 状态机：`(P, Z, t_k)` + 事件优先级
- 事件回流：出界/角球/任意球/点球/射门分解

与 `datamodel_state_machine_v2.md` 对齐要点：

- 第3章：基础参数、策略关系、路线倾向
- 第4章：对抗与过程公式
- 第6章：事件回流规则
- 第8章：90分钟流程控制
- 第9章：AIGC输入输出

---

## 4.4 展示层（Presentation Layer）

统一采用 Pygame 单窗口分区渲染：

- 左侧：球场、球员、足球、区域/球权叠层
- 右上：摄像头帧 + 手部骨架 + 手势状态
- 右下：战术控件、8维评分、统计、日志、AI报告弹窗
- 不再引入 Tkinter Canvas 作为显示容器

---

## 5. 关键接口与数据契约

## 5.1 手势到战术命令映射（建议）

| 手势事件 | 业务语义 | 引擎动作 |
| --- | --- | --- |
| `formation_detected` | 识别出目标阵型 | 预设 `var_form_A`，等待确认 |
| `confirm` | 确认应用 | 调用 `apply_team_adjustment("A", ...)` |
| `move_player/release` | 手动位置意图 | 记录为“临时阵型编辑态”，确认后映射到阵型或策略偏置 |

## 5.2 状态快照接口

沿用：
- `engine.get_snapshot()`
- `engine.build_a_analysis_context(...)`
- `engine.build_deepseek_prompt_payload(...)`

新增建议：
- `get_fusion_snapshot()`：增加手势状态、命令队列状态、确认锁状态

---

## 6. 实施步骤（建议3阶段）

## 6.1 阶段A：能力打通（1-2天）

- 抽离 Gesture Adapter（从 `GestureFormationApp` 中剥离识别与命令生成）
- 在 `PygameFusionApp` 增加命令接收器与状态展示
- 完成“识别阵型 -> 确认 -> 应用引擎”闭环

交付物：
- 可运行融合入口（单进程）
- 基础日志可追溯

## 6.2 阶段B：流程稳定（1-2天）

- 增加去抖与冲突处理（例如确认前冻结/排队）
- 对齐时间窗配额逻辑与手势触发逻辑
- 增加失败兜底（手势丢失、摄像头异常、命令过期）

交付物：
- 稳定运行的融合版
- 回归用例集（关键路径）

## 6.3 阶段C：体验优化（2-3天）

- 优化看板中的手势反馈、动画同步、文案可解释性
- 优化中场/终场报告呈现样式与可读性
- 输出产品演示流程脚本

交付物：
- Demo版融合产品
- 对外展示材料（流程图+关键截图）

---

## 7. 风险与解决策略

## 7.1 主循环与线程协同风险

- 风险：Pygame 渲染、摄像头采集、LLM请求抢占主线程导致卡顿
- 策略：Pygame 主循环固定，摄像头与LLM全部后台线程化；主线程仅渲染与事件分发

## 7.2 手势误识别导致错误应用

- 风险：错误阵型确认
- 策略：双阶段确认（识别态 -> OK确认态）+ 超时回退

## 7.3 状态不一致

- 风险：手势端与引擎端阵型不同步
- 策略：引擎为唯一真相源，手势端仅发意图不持久化比赛状态

## 7.4 模型分析时延

- 风险：中场停顿过长
- 策略：上下文压缩（`timeline_digest`）+ 异步调用 + 弹窗占位提示

---

## 8. 验收标准（融合版）

1. 单入口启动后，手势识别与模拟器同时工作且不互相阻塞。  
2. 手势确认后，A队阵型变更可在看板与引擎中同步可见。  
3. 时间窗配额限制对手势与控件操作都生效。  
4. 比赛推进过程中，日志、统计、区域图与状态机结果一致。  
5. 45:00/90:00 可自动生成并展示战术分析报告（15秒自动隐藏）。  

---

## 9. 里程碑建议

- **M1（融合可用）**：手势命令驱动引擎成功，核心流程跑通  
- **M2（融合稳定）**：异常处理完善，回归测试通过  
- **M3（融合可演示）**：交互体验优化，支持完整产品演示  

---

## 10. 本文档输出结论

建议采用“Pygame 主界面 + 手势适配器后台线程 + 状态机单一真相源”的融合路线。该路线符合当前交互目标，且与既有手势UI代码兼容度最高，可在短周期内实现从“双原型并行”到“单产品闭环”的过渡。

---

## 11. 融合版界面字段清单（V1）

以下字段用于融合版单界面（左球场 + 右摄像头/看板）最小可用集。

| 区域 | 字段ID（建议） | 字段名称 | 展示类型 | 备注 |
| --- | --- | --- | --- | --- |
| 左侧球场 | `pitch_bg` | 球场背景（`ground.jpg`） | 图像 | 保持原始长宽比 |
| 左侧球场 | `ball_marker` | 足球位置（`players/ball.png`） | 图像 | 由 7x3 区域映射坐标 |
| 左侧球场 | `players_a` | A队11人头像 | 图像组 | 支持抓取/移动/确认 |
| 左侧球场 | `players_b` | B队11人头像 | 图像组 | 仅自动变阵，不可抓取 |
| 左侧球场 | `overlay_possession` | 当前控球方 | 文本角标 | A/B 队中文显示 |
| 左侧球场 | `overlay_zone` | 当前区域 | 文本角标 | 7x3 中文区名 |
| 左侧球场 | `overlay_event` | 最新事件 | 文本角标 | 推进/射门/定位球等 |
| 右上 | `camera_feed` | 摄像头画面 | 视频 | OpenCV 帧 |
| 右上 | `hand_skeleton` | 手部骨架叠加 | 图形叠加 | MediaPipe 关键点 |
| 右上 | `gesture_state` | 手势状态 | 文本 | 未检测/抓取/移动/放开/确认 |
| 右下-比赛 | `scoreboard` | 比分 | 文本 | A x:y B |
| 右下-比赛 | `match_clock` | 比赛时间 | 文本 | `mm:ss / 90:00` |
| 右下-比赛 | `phase_window` | 半场+时间窗 | 文本 | 上/下半场 + `window` |
| 右下-比赛 | `team_a_form` | A队当前阵型 | 文本 | 含最近一次变更时间 |
| 右下-比赛 | `team_b_form` | B队当前阵型 | 文本 | 自动调整结果 |
| 右下-比赛 | `team_a_strategy` | A队当前策略（三组） | 单选组+文本 | 高度/节奏/主攻通道 |
| 右下-比赛 | `team_b_strategy` | B队当前策略（三组） | 文本 | 自动调整结果 |
| 右下-比赛 | `quota_a` | A队调整配额 | 文本 | 阵型剩余/策略剩余 |
| 右下-比赛 | `quota_b` | B队调整配额 | 文本 | 阵型剩余/策略剩余 |
| 右下-评分 | `feature_bars_a` | A队8维百分比 | 进度条组 | 加权值主条 |
| 右下-评分 | `feature_delta_a` | A队8维增量 | 彩色小条 | 红=正，绿=负，0不显示 |
| 右下-评分 | `route_weights_a` | A队路线权重 | 文本/条形 | 左/中/右 |
| 右下-日志 | `event_log` | 比赛过程日志 | 滚动文本 | 可筛选类别 |
| 右下-统计 | `pro_stats` | 专业统计面板 | 表格/文本 | 射门、射正、失误、反击等 |

---

## 12. 状态映射表（UI字段 -> 引擎变量）

说明：

- 引擎变量以 `match_state_machine_simulator.py` 中 `MatchSimulatorEngine.get_snapshot()` 返回结构为准。
- 手势相关字段来自融合控制器（`FusionController` / `Gesture Adapter`）的运行态。
- `Derived` 表示前端显示层计算字段，不直接存储在引擎状态中。

| UI字段ID | 数据来源 | 引擎变量/状态路径 | 计算方式 | 更新触发 |
| --- | --- | --- | --- | --- |
| `scoreboard` | Engine Snapshot | `snap["score"] -> (a,b)` | `A a:b B` 格式化 | 每次推进/重置/调整后 |
| `match_clock` | Engine Snapshot | `snap["match_seconds"]` | `format_match_time()` | 每次推进后 |
| `phase_window` | Engine Snapshot | `snap["window"]` + `snap["match_seconds"]` | `window+1/6` + 上/下半场判定 | 每次推进后 |
| `overlay_possession` | Engine Snapshot | `snap["possession"]` | `TEAM_CN` 中文映射 | 每次推进后 |
| `overlay_zone` | Engine Snapshot | `snap["zone"]` | `zone_tuple_to_cn()` | 每次推进后 |
| `overlay_event` | Engine Snapshot | `snap["last_event"]` | 直接显示 | 每次推进后 |
| `ball_marker` | Engine Snapshot + UI映射 | `snap["zone"]`, `snap["possession"]` | `zone -> absolute_zone -> canvas(x,y)` | 每次推进后 |
| `team_a_form` | Engine Snapshot | `snap["A"]["tactics"].formation` | 直接显示 | 调整应用后/推进后 |
| `team_b_form` | Engine Snapshot | `snap["B"]["tactics"].formation` | 直接显示 | 自动调整后/推进后 |
| `team_a_strategy` | Engine Snapshot | `snap["A"]["tactics"].strategy_height/tempo/channel` | 三组拼接 | 调整应用后/推进后 |
| `team_b_strategy` | Engine Snapshot | `snap["B"]["tactics"].strategy_height/tempo/channel` | 三组拼接 | 自动调整后/推进后 |
| `quota_a` | Engine Snapshot | `snap["A"]["quota"].formation_left/strategy_left` | 文本格式化 | 调整应用后/窗口切换 |
| `quota_b` | Engine Snapshot | `snap["B"]["quota"].formation_left/strategy_left` | 文本格式化 | 自动调整后/窗口切换 |
| `feature_bars_a` | Engine Snapshot | `snap["A"]["weighted"][FEATURE_KEY]` | 百分比条（0-1 -> 0-100%） | 调整应用后/推进后 |
| `feature_delta_a` | Derived | `snap["A"]["weighted"] - snap["A"]["base"]` | 红正绿负零隐藏 | 调整应用后 |
| `route_weights_a` | Engine Snapshot | `snap["A"]["route"] -> (L,C,R)` | 左/中/右显示 | 调整应用后/推进后 |
| `pro_stats` | Engine Snapshot | `snap["stats"]["A"]`, `snap["stats"]["B"]` | 统计项格式化 | 每次推进后 |
| `event_log` | Engine Report | `StepReport.summary`, `StepReport.detail_lines` | 按类别过滤后渲染 | 每次推进后 |
| `players_a` | Gesture Runtime + Tactics | `gesture_state.a_player_positions` | 头像坐标渲染 | 每帧 + 动画tick |
| `players_b` | Auto-tactic Runtime | `fusion_state.b_target_formation` + 动画状态 | 头像坐标渲染 | 自动变阵调度tick |
| `camera_feed` | Gesture Runtime | `gesture_state.camera_frame` | BGR->RGB + 贴图 | 每帧 |
| `hand_skeleton` | Gesture Runtime | `gesture_state.hand_landmarks` | 关键点连线渲染 | 每帧 |
| `gesture_state` | Gesture Runtime | `gesture_state.action` | 中文状态映射 | 每帧 |

### 12.1 融合层建议新增状态（非引擎内）

| 状态名 | 类型 | 作用 |
| --- | --- | --- |
| `fusion_state.pending_a_confirm` | `bool` | A队手势调整是否待确认 |
| `fusion_state.b_window_plan` | `dict` | B队当前15分钟窗口的随机调整计划 |
| `fusion_state.b_pending_animation` | `list` | A队确认前累计的B队待执行动画队列 |
| `fusion_state.zone_to_canvas_map` | `dict` | 7x3区域到球场像素坐标映射表 |
| `fusion_state.log_filters` | `set` | 日志类别筛选（事件/概率/结果等） |
