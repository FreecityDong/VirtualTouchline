import json
import math
import os
import random
import re
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import pygame
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

import gesture_formation_interaction as gfi
from match_state_machine_simulator import (
    CHANNEL_STRATEGIES,
    FEATURE_KEYS,
    FEATURE_LABELS,
    FORMATIONS,
    HEIGHT_STRATEGIES,
    LANE_CN,
    TEAM_CN,
    TEMPO_STRATEGIES,
    MatchSimulatorEngine,
    format_match_time,
    zone_tuple_to_cn,
)


class FusionPygameApp(gfi.GestureFormationApp):
    """
    融合入口（阶段1）：
    - 复用 gesture_formation_interaction.py 的球场、球员、手势交互
    - 接入 MatchSimulatorEngine 作为比赛状态机
    - 以 pygame 作为唯一主界面
    """

    AUTO_TOTAL_REAL_SECONDS = 5 * 60
    HALF_TIME_SECONDS = 45 * 60
    MATCH_END_SECONDS = 90 * 60
    LOG_MAX_LINES = 180
    BALL_ICON_SIZE = 24
    PITCH_BOTTOM_CONTROL_RESERVE = 90
    HAND_MAP_X_MIN = 0.06
    HAND_MAP_X_MAX = 0.94
    HAND_MAP_X_GAIN = 1.12
    HAND_MAP_Y_MIN = 0.04
    HAND_MAP_Y_MAX = 0.96
    HAND_MAP_Y_GAIN = 1.08
    CURSOR_SMOOTH_ALPHA_IDLE = 0.38
    CURSOR_SMOOTH_ALPHA_GRAB = 0.18
    CURSOR_SMOOTH_ALPHA_DRAG = 0.45
    CURSOR_DEADZONE_PX_GRAB = 8.0
    CURSOR_MAX_STEP_PX_GRAB = 26.0
    CURSOR_MAX_STEP_PX_DRAG = 80.0
    STRATEGY_SELECT_HOLD_SECONDS = 0.5
    STRATEGY_CONFIRM_OK_SECONDS = 1.0
    REPORT_POPUP_SHOW_SECONDS = 30.0
    REPORT_OK_CLOSE_SECONDS = 1.0
    LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL = "deepseek-v3.2"
    LLM_DEFAULT_API_KEY = "sk-bac18f9772b14f35ad33a8e53e5d9809"

    PANEL_BG = (24, 24, 24)
    PANEL_CARD = (36, 36, 36)
    PANEL_BORDER = (70, 70, 70)
    PANEL_TEXT = (230, 230, 230)
    PANEL_MUTED = (170, 170, 170)
    PANEL_ACCENT = (120, 200, 255)
    GREEN = (76, 200, 120)
    RED = (220, 100, 100)

    ZONE_ROW_ORDER = ["A_PA", "A_BA", "A_DZ", "MC", "B_DZ", "B_BA", "B_PA"]
    ZONE_ROW_X_RATIO = {
        "A_PA": 0.10,
        "A_BA": 0.19,
        "A_DZ": 0.31,
        "MC": 0.50,
        "B_DZ": 0.69,
        "B_BA": 0.81,
        "B_PA": 0.90,
    }
    LANE_Y_RATIO = {"L": 0.25, "C": 0.50, "R": 0.75}

    A_FORM_PRESETS = ["4-4-2", "4-2-3-1", "4-3-3", "3-5-2", "5-4-1"]

    def __init__(self) -> None:
        super().__init__()

        # 关闭基类中B队“每10秒自动换阵”的逻辑，改由状态机时间窗调度
        self.team_b_auto_switch_interval = 10_000.0

        self.engine = MatchSimulatorEngine()
        self.auto_slot_seconds = self._compute_auto_slot_seconds()
        self.auto_running = False
        self.halftime_paused = False
        self.last_auto_tick_time = time.time()

        self.last_report = None
        self.log_lines: List[str] = []
        self.click_regions: List[Dict[str, object]] = []
        self.scroll_offset = 0
        self.scroll_max = 0
        self.scroll_viewport: Optional[pygame.Rect] = None
        self.camera_surface_raw: Optional[pygame.Surface] = None
        self.camera_available = False
        self.last_pitch_image_rect: Optional[pygame.Rect] = None
        self.camera_fail_streak = 0
        self.camera_fail_limit = 24
        self.camera_reinit_interval = 2.0
        self.last_camera_reinit_attempt = 0.0
        self.last_good_frame_time = 0.0
        self.gesture_interval_seconds = 1.0 / 30.0
        self.last_gesture_infer_time = 0.0
        self.camera_offline_reported = False
        self.camera_indices = [0, 1, 2]
        self._camera_lock = threading.Lock()
        self._camera_stop_event = threading.Event()
        self._camera_thread: Optional[threading.Thread] = None
        self._camera_cap = self.cap
        self.cap = None
        self._latest_camera_frame = None
        self._latest_camera_ts = 0.0
        self.live_recognized_formation = "未识别"
        self.live_matched_formation = "未匹配"
        self.preview_formation_engine = "4-4-2"
        self.team_event_traces: Dict[str, List[str]] = {"A": [], "B": []}
        self.team_zone_view: Dict[str, Tuple[str, str]] = {"A": ("MC", "C"), "B": ("MC", "C")}
        self.team_route_view: Dict[str, str] = {"A": "中路", "B": "中路"}
        self.pitch_control_regions: List[Dict[str, object]] = []
        self.pitch_control_bar_rect: Optional[pygame.Rect] = None
        self.pitch_control_hold_seconds = 1.0
        self.pitch_control_hold_start_time: Optional[float] = None
        self.pitch_control_hold_target: Optional[str] = None
        self.pitch_control_wait_release = False
        self.cursor_filtered_x: Optional[float] = None
        self.cursor_filtered_y: Optional[float] = None
        self.strategy_hold_start_time: Optional[float] = None
        self.strategy_hold_target_value: Optional[str] = None
        self.strategy_hold_wait_release = False
        self.strategy_confirm_pending = False
        self.halftime_report_generated = False
        self.fulltime_report_generated = False
        self.report_cache: Dict[str, str] = {}
        self.report_generating: Dict[str, bool] = {"halftime": False, "fulltime": False}
        self.report_auto_open_pending: Dict[str, bool] = {"halftime": False, "fulltime": False}
        self.report_result_queue: List[Tuple[str, str, str]] = []
        self.report_result_lock = threading.Lock()
        self.report_popup_visible = False
        self.report_popup_title = ""
        self.report_popup_content = ""
        self.report_popup_type: Optional[str] = None
        self.report_popup_auto_close_at: Optional[float] = None
        self.report_popup_ok_hold_start: Optional[float] = None
        self.report_popup_wait_release = False

        self.a_strategy_select = {
            "height": "默认平衡",
            "tempo": "默认平衡",
            "channel": "默认平衡",
        }

        self.height_options = list(HEIGHT_STRATEGIES.keys())
        self.tempo_options = list(TEMPO_STRATEGIES.keys())
        self.channel_options = list(CHANNEL_STRATEGIES.keys())

        self.ui_to_engine_form, self.engine_to_ui_form = self._build_formation_mappers()
        self.selected_formation_a = self._engine_to_ui("4-4-2")
        self.selected_formation_b = self._engine_to_ui("4-4-2")
        self.create_players()

        self.ball_icon = self._load_ball_icon()
        self.b_window_plan: Dict[int, List[Dict[str, object]]] = {}
        self._ensure_b_window_plan(0)

        self._append_structured_log("系统初始化：Pygame主界面 + 状态机引擎")
        self._append_structured_log(
            f"自动推进速度：每{self.auto_slot_seconds:.2f}秒推进{self.engine.STEP_SECONDS}秒赛时（整场约{self.AUTO_TOTAL_REAL_SECONDS / 60:.1f}分钟）"
        )
        self._sync_a_to_engine(force=True)
        self._sync_b_to_engine(force=True)
        self.preview_formation_engine = self.engine.team_a.formation
        self._start_camera_thread()

    def _load_ball_icon(self) -> Optional[pygame.Surface]:
        try:
            img = pygame.image.load("players/ball.png").convert_alpha()
            return pygame.transform.smoothscale(img, (self.BALL_ICON_SIZE, self.BALL_ICON_SIZE))
        except Exception:
            return None

    def _compute_auto_slot_seconds(self) -> float:
        total_steps = max(1, math.ceil(self.engine.MATCH_DURATION_SECONDS / self.engine.STEP_SECONDS))
        slot = self.AUTO_TOTAL_REAL_SECONDS / total_steps
        return max(0.10, slot)

    def _get_pitch_rects(self):
        """为左侧按钮预留底部空间，保证按钮可放在球场图下方。"""
        sw, sh = self.screen.get_size()
        sidebar_w = max(320, int(sw * 0.28))
        pitch_h = max(220, sh - 20 - self.PITCH_BOTTOM_CONTROL_RESERVE)
        pitch_rect = pygame.Rect(10, 10, sw - sidebar_w - 20, pitch_h)

        img_w, img_h = self.ground_size
        margin = 10
        fit_x, fit_y, fit_w, fit_h = gfi.compute_image_rect(
            pitch_rect.width, pitch_rect.height, img_w, img_h, margin
        )
        image_rect = pygame.Rect(
            pitch_rect.x + fit_x,
            pitch_rect.y + fit_y,
            fit_w,
            fit_h,
        )
        return pitch_rect, image_rect

    def map_hand_to_screen(self, hand_coords):
        """
        覆盖基类映射：
        - 扩展有效归一化范围，减小边缘不可达问题
        - 轻微提高纵向增益，让光标更容易到下方按钮区
        """
        if not hand_coords:
            return None
        hand_x_norm, hand_y_norm = hand_coords

        x_span = max(1e-6, self.HAND_MAP_X_MAX - self.HAND_MAP_X_MIN)
        y_span = max(1e-6, self.HAND_MAP_Y_MAX - self.HAND_MAP_Y_MIN)
        x_norm = (hand_x_norm - self.HAND_MAP_X_MIN) / x_span
        y_norm = (hand_y_norm - self.HAND_MAP_Y_MIN) / y_span
        x_norm = (x_norm - 0.5) * self.HAND_MAP_X_GAIN + 0.5
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm * self.HAND_MAP_Y_GAIN))

        sw, sh = self.screen.get_size()
        sidebar_w = max(320, int(sw * 0.28))
        playable_left = 10
        playable_top = 10
        playable_w = sw - sidebar_w - 20
        playable_h = sh - 20

        screen_x = playable_left + playable_w * (1.0 - x_norm)  # 镜像X轴
        screen_y = playable_top + playable_h * y_norm
        return int(screen_x), int(screen_y)

    def _update_hand_cursor(self, mapped: Tuple[int, int]) -> None:
        raw_x, raw_y = float(mapped[0]), float(mapped[1])
        if self.cursor_filtered_x is None or self.cursor_filtered_y is None:
            self.cursor_filtered_x = raw_x
            self.cursor_filtered_y = raw_y
        else:
            dx = raw_x - self.cursor_filtered_x
            dy = raw_y - self.cursor_filtered_y
            dist = math.hypot(dx, dy)
            if self.is_grabbing:
                if self.dragging_player is not None:
                    if dist > self.CURSOR_MAX_STEP_PX_DRAG:
                        scale = self.CURSOR_MAX_STEP_PX_DRAG / max(1e-6, dist)
                        raw_x = self.cursor_filtered_x + dx * scale
                        raw_y = self.cursor_filtered_y + dy * scale
                    alpha = self.CURSOR_SMOOTH_ALPHA_DRAG
                else:
                    if dist <= self.CURSOR_DEADZONE_PX_GRAB:
                        raw_x = self.cursor_filtered_x
                        raw_y = self.cursor_filtered_y
                    elif dist > self.CURSOR_MAX_STEP_PX_GRAB:
                        scale = self.CURSOR_MAX_STEP_PX_GRAB / max(1e-6, dist)
                        raw_x = self.cursor_filtered_x + dx * scale
                        raw_y = self.cursor_filtered_y + dy * scale
                    alpha = self.CURSOR_SMOOTH_ALPHA_GRAB
            else:
                alpha = self.CURSOR_SMOOTH_ALPHA_IDLE
            self.cursor_filtered_x = self.cursor_filtered_x + alpha * (raw_x - self.cursor_filtered_x)
            self.cursor_filtered_y = self.cursor_filtered_y + alpha * (raw_y - self.cursor_filtered_y)

        self.hand_x = int(round(self.cursor_filtered_x))
        self.hand_y = int(round(self.cursor_filtered_y))

    @staticmethod
    def _strategy_group_label(group: str) -> str:
        return {"height": "高度", "tempo": "节奏", "channel": "通道"}.get(group, group)

    def _handle_strategy_gesture_selection(self) -> None:
        strategy_items = [item for item in self.click_regions if str(item.get("kind", "")) == "strategy"]
        if not strategy_items:
            self.strategy_hold_start_time = None
            self.strategy_hold_target_value = None
            return
        if self.strategy_hold_wait_release:
            if not self.is_grabbing:
                self.strategy_hold_wait_release = False
            return
        if not self.hand_detected or self.dragging_player is not None:
            self.strategy_hold_start_time = None
            self.strategy_hold_target_value = None
            return

        hovered = None
        hand_pos = (self.hand_x, self.hand_y)
        hover_padding = 10
        for item in strategy_items:
            rect = item["rect"]
            if rect.inflate(hover_padding * 2, hover_padding * 2).collidepoint(hand_pos):
                hovered = item
                break

        if (hovered is None) or (not self.is_grabbing):
            self.strategy_hold_start_time = None
            self.strategy_hold_target_value = None
            return

        hovered_value = str(hovered["value"])
        now = time.time()
        if self.strategy_hold_target_value != hovered_value:
            self.strategy_hold_target_value = hovered_value
            self.strategy_hold_start_time = now
            return
        if self.strategy_hold_start_time is None:
            self.strategy_hold_start_time = now
            return

        if now - self.strategy_hold_start_time < self.STRATEGY_SELECT_HOLD_SECONDS:
            return

        self.strategy_hold_wait_release = True
        self.strategy_hold_start_time = None
        self.strategy_hold_target_value = None
        if not bool(hovered.get("enabled", True)):
            self.confirmation_message = "策略选择失败：当前窗口配额不足"
            self.confirmation_timer = 120
            return

        group, opt = hovered_value.split(":", 1)
        old = self.a_strategy_select[group]
        if old == opt:
            return
        self.a_strategy_select[group] = opt
        self.strategy_confirm_pending = True
        group_cn = self._strategy_group_label(group)
        self.confirmation_message = f"A队策略待确认：{group_cn}->{self._strategy_short_name(group, opt)}（OK手势1秒）"
        self.confirmation_timer = 160

    def _confirm_pending_strategy_with_ok(self) -> None:
        ta = self.engine.team_a
        if not self.strategy_confirm_pending:
            return
        if not (self.hand_detected and self.hand_keypoints and self.hand_keypoints.get("is_ok_gesture", False)):
            self.ok_hold_start_time = None
            self.ok_hold_elapsed = 0.0
            self.ok_confirmed_wait_release = False
            return
        now = time.time()
        if self.ok_confirmed_wait_release:
            return
        if self.ok_hold_start_time is None:
            self.ok_hold_start_time = now
            self.ok_hold_elapsed = 0.0
            return

        self.ok_hold_elapsed = now - self.ok_hold_start_time
        if self.ok_hold_elapsed < self.STRATEGY_CONFIRM_OK_SECONDS:
            return

        target_state = (
            self.a_strategy_select["height"],
            self.a_strategy_select["tempo"],
            self.a_strategy_select["channel"],
        )
        before_state = (ta.strategy_height, ta.strategy_tempo, ta.strategy_channel)
        self._sync_a_to_engine(force=False)
        after_state = (ta.strategy_height, ta.strategy_tempo, ta.strategy_channel)
        if after_state == target_state and after_state != before_state:
            self.confirmation_message = "A队策略确认成功"
            self.confirmation_timer = 140
            self.strategy_confirm_pending = False
        elif target_state == before_state:
            self.confirmation_message = "A队策略未变化"
            self.confirmation_timer = 100
            self.strategy_confirm_pending = False
        else:
            # _sync_a_to_engine 内部会在拒绝时回滚到引擎状态
            self.confirmation_message = "A队策略确认失败：配额不足"
            self.confirmation_timer = 140
            self.strategy_confirm_pending = False

        self.last_ok_gesture_time = now
        self.ok_confirmed_wait_release = True
        self.ok_hold_start_time = None
        self.ok_hold_elapsed = 0.0

    @staticmethod
    def _extract_formation_digits(name: str) -> Optional[str]:
        if not name:
            return None
        m = re.search(r"(\d-\d(?:-\d){1,3})", name)
        return m.group(1) if m else None

    def _canonical_engine_formation(self, name: str) -> str:
        if not name:
            return "4-4-2"
        if name in FORMATIONS:
            return name

        normalized = (
            name.replace("（", "(")
            .replace("）", ")")
            .replace(" ", "")
            .replace("：", ":")
        )
        if normalized in FORMATIONS:
            return normalized

        digits = self._extract_formation_digits(normalized)
        if "链式防守" in normalized:
            return "5-3-2链式防守"
        if "伪九号" in normalized:
            return "4-3-3伪九号"
        if "无锋阵" in normalized:
            return "4-6-0"
        if normalized.startswith("WM") and digits:
            wm = f"WM({digits})"
            if wm in FORMATIONS:
                return wm
        if digits and digits in FORMATIONS:
            return digits

        for key in FORMATIONS:
            if normalized.startswith(key):
                return key
        return "4-4-2"

    def _build_formation_mappers(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        ui_to_engine: Dict[str, str] = {}
        engine_to_ui: Dict[str, str] = {}
        for ui_name in self.formation_names:
            eng = self._canonical_engine_formation(ui_name)
            ui_to_engine[ui_name] = eng
            engine_to_ui.setdefault(eng, ui_name)

        # 兜底：若某些引擎阵型在md中没有同名条目，直接使用引擎名
        for eng in FORMATIONS.keys():
            engine_to_ui.setdefault(eng, eng)
        return ui_to_engine, engine_to_ui

    def _ui_to_engine(self, ui_name: str) -> str:
        return self.ui_to_engine_form.get(ui_name, self._canonical_engine_formation(ui_name))

    def _engine_to_ui(self, engine_name: str) -> str:
        return self.engine_to_ui_form.get(engine_name, engine_name)

    def _append_log(self, text: str) -> None:
        self.log_lines.append(text)
        if len(self.log_lines) > self.LOG_MAX_LINES:
            self.log_lines = self.log_lines[-self.LOG_MAX_LINES :]

    def _append_structured_log(
        self,
        event_text: str,
        time_text: Optional[str] = None,
        possession: Optional[str] = None,
        zone_cn: Optional[str] = None,
        route: Optional[str] = None,
    ) -> None:
        snap = self.engine.get_snapshot()
        t = time_text or format_match_time(int(snap["match_seconds"]))
        p = TEAM_CN.get(possession or str(snap["possession"]), possession or str(snap["possession"]))
        z = zone_cn or zone_tuple_to_cn(snap["zone"])
        r = route or str(snap["last_route"])
        line = f"{t} | {p} | {z} | {r} | {event_text}"
        self._append_log(line)

    def _refresh_live_formation_match(self) -> None:
        adjusting = (self.dragging_player is not None and self.dragging_player.team == "A") or self.a_position_dirty_pending_ok
        if not adjusting:
            self.live_recognized_formation = "未识别"
            self.live_matched_formation = "未匹配"
            self.preview_formation_engine = self.engine.team_a.formation
            return
        recognized = self.recognize_formation()
        self.live_recognized_formation = recognized or "未识别"
        matched = self.find_closest_standard_formation(recognized) if recognized else None
        if matched:
            self.live_matched_formation = matched
            self.preview_formation_engine = self._ui_to_engine(matched)
        else:
            self.live_matched_formation = "未匹配"
            self.preview_formation_engine = self.engine.team_a.formation

    def _update_team_event_views_from_last_step(self) -> None:
        if not self.engine.event_history:
            return
        ev = self.engine.event_history[-1]
        atk = str(ev.get("attack_team", "A"))
        poss_before = str(ev.get("possession_before", atk))
        poss_after = str(ev.get("possession_after", poss_before))
        zone_before_cn = str(ev.get("zone_before_cn", ""))
        zone_after_cn = str(ev.get("zone_after_cn", ""))
        event_text = str(ev.get("event", ""))

        line = f"{zone_before_cn}->{event_text}"
        if poss_before != poss_after:
            # 球权交换时，清除失误方事件，在新控球方侧显示
            self.team_event_traces[poss_before] = []
            self.team_event_traces[poss_after].append(line)
            self.team_event_traces[poss_after] = self.team_event_traces[poss_after][-5:]
        else:
            self.team_event_traces[atk].append(line)
            self.team_event_traces[atk] = self.team_event_traces[atk][-5:]

        zone_after_raw = str(ev.get("zone_after", "MC_C"))
        try:
            level, lane = zone_after_raw.split("_", 1)
            self.team_zone_view[poss_after] = (level, lane)
        except Exception:
            pass
        self.team_route_view[poss_after] = str(self.engine.state.last_route)

        # 为非控球方提供默认展示：使用当前状态镜像到对方半场
        non = "B" if poss_after == "A" else "A"
        snap = self.engine.get_snapshot()
        other_zone = snap["zone"]
        self.team_zone_view[non] = other_zone

    @staticmethod
    def _to_absolute_zone_key(zone: Tuple[str, str], possession: str) -> Optional[str]:
        level, lane = zone
        if lane not in {"L", "C", "R"}:
            return None
        if possession == "A":
            level_map = {"DZ": "A_DZ", "MC": "MC", "AT": "B_DZ", "BA": "B_BA", "PA": "B_PA"}
        else:
            level_map = {"DZ": "B_DZ", "MC": "MC", "AT": "A_DZ", "BA": "A_BA", "PA": "A_PA"}
        abs_level = level_map.get(level)
        if abs_level is None:
            return None
        return f"{abs_level}_{lane}"

    def _zone_point(self, image_rect: pygame.Rect, zone: Tuple[str, str], possession: str) -> Optional[Tuple[int, int]]:
        key = self._to_absolute_zone_key(zone, possession)
        if not key:
            return None
        level, lane = key.rsplit("_", 1)
        x_ratio = self.ZONE_ROW_X_RATIO.get(level)
        y_ratio = self.LANE_Y_RATIO.get(lane)
        if x_ratio is None or y_ratio is None:
            return None
        x = int(image_rect.left + image_rect.width * x_ratio)
        y = int(image_rect.top + image_rect.height * y_ratio)
        return x, y

    def _draw_ball_marker(self, image_rect: pygame.Rect) -> None:
        snap = self.engine.get_snapshot()
        point = self._zone_point(image_rect, snap["zone"], snap["possession"])
        if point is None:
            return
        x, y = point
        pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 14, 2)
        if self.ball_icon is not None:
            rect = self.ball_icon.get_rect(center=(x, y))
            self.screen.blit(self.ball_icon, rect)
        else:
            pygame.draw.circle(self.screen, (255, 245, 180), (x, y), 6)

    def _ensure_b_window_plan(self, window_idx: int) -> None:
        if window_idx in self.b_window_plan:
            return
        start = window_idx * self.engine.WINDOW_SECONDS
        end = min(start + self.engine.WINDOW_SECONDS, self.engine.MATCH_DURATION_SECONDS)
        points = list(range(start + self.engine.STEP_SECONDS, end + 1, self.engine.STEP_SECONDS))
        if not points:
            self.b_window_plan[window_idx] = []
            return

        form_n = random.randint(0, 1)
        strat_n = random.randint(0, 2)
        events: List[Dict[str, object]] = []
        if form_n > 0:
            t = random.choice(points)
            events.append({"time": t, "kind": "formation", "done": False})
        for _ in range(strat_n):
            t = random.choice(points)
            group = random.choice(["height", "tempo", "channel"])
            events.append({"time": t, "kind": "strategy", "group": group, "done": False})
        events.sort(key=lambda x: int(x["time"]))
        self.b_window_plan[window_idx] = events
        self._append_structured_log(f"B队计划：时间窗{window_idx + 1}/6 阵型{form_n}次 策略{strat_n}次")

    def _pick_new_b_formation(self) -> str:
        current = self.engine.team_b.formation
        pool = [x for x in FORMATIONS.keys() if x != current]
        if not pool:
            return current
        return random.choice(pool)

    def _pick_new_b_strategy_value(self, group: str) -> str:
        tb = self.engine.team_b
        if group == "height":
            current = tb.strategy_height
            opts = [x for x in self.height_options if x != current]
        elif group == "tempo":
            current = tb.strategy_tempo
            opts = [x for x in self.tempo_options if x != current]
        else:
            current = tb.strategy_channel
            opts = [x for x in self.channel_options if x != current]
        return random.choice(opts) if opts else current

    def _execute_due_b_events(self) -> None:
        w = self.engine.state.current_window
        self._ensure_b_window_plan(w)
        now_sec = self.engine.state.match_seconds
        events = self.b_window_plan.get(w, [])
        for ev in events:
            if ev.get("done"):
                continue
            if now_sec < int(ev["time"]):
                continue

            if ev["kind"] == "formation":
                target_eng = self._pick_new_b_formation()
                tb = self.engine.team_b
                ok, msgs = self.engine.apply_team_adjustment(
                    "B",
                    target_eng,
                    tb.strategy_height,
                    tb.strategy_tempo,
                    tb.strategy_channel,
                )
                if ok:
                    ui_target = self._engine_to_ui(target_eng)
                    self.adjust_formation_to_standard(ui_target, team_name="B")
                    self._append_structured_log(
                        event_text=f"B队自动阵型调整 -> {target_eng}",
                        time_text=format_match_time(now_sec),
                    )
                for m in msgs:
                    self._append_structured_log(f"B队自动调整: {m}", time_text=format_match_time(now_sec))
            elif ev["kind"] == "strategy":
                group = str(ev.get("group", "height"))
                tb = self.engine.team_b
                nh, nt, nc = tb.strategy_height, tb.strategy_tempo, tb.strategy_channel
                val = self._pick_new_b_strategy_value(group)
                if group == "height":
                    nh = val
                elif group == "tempo":
                    nt = val
                else:
                    nc = val
                ok, msgs = self.engine.apply_team_adjustment("B", tb.formation, nh, nt, nc)
                if ok:
                    self._append_structured_log(
                        event_text=f"B队自动策略调整({group}) -> {val}",
                        time_text=format_match_time(now_sec),
                    )
                for m in msgs:
                    self._append_structured_log(f"B队自动调整: {m}", time_text=format_match_time(now_sec))

            ev["done"] = True

    def _sync_a_to_engine(self, force: bool = False) -> None:
        ta = self.engine.team_a
        target_form = self._ui_to_engine(self.selected_formation_a)
        h = self.a_strategy_select["height"]
        t = self.a_strategy_select["tempo"]
        c = self.a_strategy_select["channel"]
        if (
            not force
            and target_form == ta.formation
            and h == ta.strategy_height
            and t == ta.strategy_tempo
            and c == ta.strategy_channel
        ):
            return
        ok, msgs = self.engine.apply_team_adjustment("A", target_form, h, t, c)
        for m in msgs:
            self._append_structured_log(f"A队调整: {m}")
        if not ok:
            # 引擎拒绝时回滚面板选择
            self.a_strategy_select["height"] = ta.strategy_height
            self.a_strategy_select["tempo"] = ta.strategy_tempo
            self.a_strategy_select["channel"] = ta.strategy_channel

    def _apply_a_formation_only_to_engine(self, target_ui_formation: str) -> bool:
        """
        仅申请A队阵型变更到引擎（不触发任何可视化调整）。
        返回值表示“阵型是否在引擎中生效”。
        """
        ta = self.engine.team_a
        target_form = self._ui_to_engine(target_ui_formation)
        if target_form == ta.formation:
            return True
        ok, msgs = self.engine.apply_team_adjustment(
            "A",
            target_form,
            ta.strategy_height,
            ta.strategy_tempo,
            ta.strategy_channel,
        )
        for m in msgs:
            self._append_structured_log(f"A队调整: {m}")
        if not ok:
            return False
        return ta.formation == target_form

    def _sync_b_to_engine(self, force: bool = False) -> None:
        tb = self.engine.team_b
        target_form = self._ui_to_engine(self.selected_formation_b)
        if (not force) and target_form == tb.formation:
            return
        ok, msgs = self.engine.apply_team_adjustment(
            "B", target_form, tb.strategy_height, tb.strategy_tempo, tb.strategy_channel
        )
        for m in msgs:
            self._append_structured_log(f"B队同步: {m}")
        if ok:
            self.selected_formation_b = self._engine_to_ui(self.engine.team_b.formation)

    def _manual_step(self) -> None:
        before_sec = self.engine.state.match_seconds
        report = self.engine.step_15min()
        self.last_report = report
        if self.engine.event_history:
            ev = self.engine.event_history[-1]
            self._append_structured_log(
                event_text=str(ev.get("event", self.engine.state.last_event)),
                time_text=str(ev.get("time", format_match_time(self.engine.state.match_seconds))),
                possession=str(ev.get("possession_after", self.engine.state.possession)),
                zone_cn=str(ev.get("zone_after_cn", zone_tuple_to_cn(self.engine.state.zone))),
                route=self.engine.state.last_route,
            )
            self._update_team_event_views_from_last_step()
        else:
            self._append_structured_log(self.engine.state.last_event)

        self._execute_due_b_events()
        self._ensure_b_window_plan(self.engine.state.current_window)

        sec = self.engine.state.match_seconds
        self._maybe_trigger_phase_reports(before_sec, sec)
        if sec >= self.HALF_TIME_SECONDS and not self.halftime_paused and sec < self.MATCH_END_SECONDS:
            self.halftime_paused = True
            self.auto_running = False
            self._append_structured_log("中场到达，自动推进暂停（按空格继续）")
        if sec >= self.MATCH_END_SECONDS:
            self.auto_running = False
            self._append_structured_log("比赛结束")

    @staticmethod
    def _sanitize_llm_user_prompt(user_prompt: Dict[str, object]) -> Dict[str, object]:
        cleaned = dict(user_prompt)
        cleaned.pop("output_json_schema", None)
        requirements = cleaned.get("requirements", [])
        if not isinstance(requirements, list):
            requirements = [str(requirements)]
        requirements.extend(
            [
                "请输出中文“战术分析报告”，不要输出JSON，不要输出Markdown代码块。",
                "报告必须使用多行结构，并保留缩进。",
                "禁止出现概率、百分比、命中率、P_等概率化字段表达。",
                "字数控制在260-420字。",
            ]
        )
        cleaned["requirements"] = requirements
        return cleaned

    def _build_llm_messages(self, report_type: str) -> List[Dict[str, str]]:
        payload = self.engine.build_deepseek_prompt_payload(
            report_type=report_type,
            include_full_timeline=True,
            timeline_scope="all",
            compact_for_llm=True,
            max_timeline_lines=(70 if report_type == "halftime" else 100),
        )
        system_prompt = str(payload["system_prompt"])
        user_prompt = self._sanitize_llm_user_prompt(payload["user_prompt"])
        return [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n"
                    "请始终使用中文，不要输出思考过程。"
                    "输出必须是可阅读的多行战术分析报告，保留换行和缩进。"
                    "不要输出任何概率化描述。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(user_prompt, ensure_ascii=False, separators=(",", ":")),
            },
        ]

    def _build_local_report_fallback(self, report_type: str, reason: str) -> str:
        try:
            context = self.engine.build_a_analysis_context(report_type=report_type, include_full_timeline=False)
            score = context.get("score_state", {})
            kpi_raw = context.get("a_phase_kpi", [])
            shots = shots_on_target = turnovers = counterattacks = 0
            if isinstance(kpi_raw, list):
                for item in kpi_raw:
                    if not isinstance(item, dict):
                        continue
                    shots += int(item.get("a_shots", 0))
                    shots_on_target += int(item.get("a_shots_on_target", 0))
                    turnovers += int(item.get("a_turnovers", 0))
                    counterattacks += int(item.get("a_counterattacks", 0))
            elif isinstance(kpi_raw, dict):
                shots = int(kpi_raw.get("a_shots", kpi_raw.get("shots", 0)))
                shots_on_target = int(kpi_raw.get("a_shots_on_target", kpi_raw.get("shots_on_target", 0)))
                turnovers = int(kpi_raw.get("a_turnovers", kpi_raw.get("turnovers", 0)))
                counterattacks = int(kpi_raw.get("a_counterattacks", kpi_raw.get("counterattacks", 0)))

            phase = "中场" if report_type == "halftime" else "终场"
            return (
                f"【{phase}战术分析报告（本地摘要）】\n"
                f"触发原因：{reason}\n"
                f"时间：{context.get('match_meta', {}).get('time', '--:--')}\n"
                f"比分：A队 {score.get('A', 0)} : {score.get('B', 0)} B队\n"
                f"当前区域：{score.get('zone_cn', '')}，控球：{TEAM_CN.get(score.get('possession', 'A'), 'A队')}\n"
                f"关键统计：射门{shots}，射正{shots_on_target}，失误{turnovers}，反击{counterattacks}\n"
                "建议：继续结合时间窗配额管理阵型与策略，优先减少失误并提升射正质量。"
            )
        except Exception as exc:
            phase = "中场" if report_type == "halftime" else "终场"
            return (
                f"【{phase}战术分析报告（本地摘要）】\n"
                f"触发原因：{reason}\n"
                f"回退生成失败：{exc}\n"
                "建议：请继续比赛，稍后重试报告。"
            )

    def _call_llm_analysis(self, report_type: str) -> str:
        if OpenAI is None:
            return self._build_local_report_fallback(report_type, "未安装 openai 依赖")
        api_key = os.getenv("DASHSCOPE_API_KEY", self.LLM_DEFAULT_API_KEY).strip()
        if not api_key:
            return self._build_local_report_fallback(report_type, "未设置 DASHSCOPE_API_KEY")
        client = OpenAI(api_key=api_key, base_url=self.LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=self.LLM_MODEL,
            messages=self._build_llm_messages(report_type),
            extra_body={"enable_thinking": False},
            stream=False,
        )
        content = ""
        if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
            content = str(resp.choices[0].message.content).strip()
        return content or self._build_local_report_fallback(report_type, "模型未返回有效内容")

    def _queue_report_result(self, report_type: str, phase_cn: str, text: str) -> None:
        with self.report_result_lock:
            self.report_result_queue.append((report_type, phase_cn, text))

    def _request_phase_report(self, report_type: str, phase_cn: str, auto_open: bool) -> None:
        if self.report_cache.get(report_type):
            if auto_open:
                self._open_report_popup(report_type, self.report_cache[report_type], phase_cn, auto_hide=True)
            return
        if self.report_generating.get(report_type, False):
            if auto_open:
                self.report_auto_open_pending[report_type] = True
                self._open_report_popup(report_type, f"正在生成{phase_cn}战术分析报告，请稍候...", phase_cn, auto_hide=False)
            return

        self.report_generating[report_type] = True
        self.report_auto_open_pending[report_type] = auto_open
        self._append_structured_log(f"[系统] 正在生成{phase_cn}战术分析报告...")
        if auto_open:
            self._open_report_popup(report_type, f"正在生成{phase_cn}战术分析报告，请稍候...", phase_cn, auto_hide=False)

        def _worker() -> None:
            try:
                text = self._call_llm_analysis(report_type)
            except Exception as exc:
                text = self._build_local_report_fallback(report_type, f"调用异常: {exc}")
            self._queue_report_result(report_type, phase_cn, text)

        threading.Thread(target=_worker, daemon=True).start()

    def _flush_report_results(self) -> None:
        queued: List[Tuple[str, str, str]] = []
        with self.report_result_lock:
            if self.report_result_queue:
                queued = self.report_result_queue[:]
                self.report_result_queue.clear()
        for report_type, phase_cn, text in queued:
            self.report_generating[report_type] = False
            self.report_cache[report_type] = text
            if report_type == "halftime":
                self.halftime_report_generated = True
            else:
                self.fulltime_report_generated = True
            self._append_structured_log(f"[系统] {phase_cn}战术分析报告已生成")
            if self.report_auto_open_pending.get(report_type, False):
                self._open_report_popup(report_type, text, phase_cn, auto_hide=True)
            self.report_auto_open_pending[report_type] = False

    def _open_report_popup(self, report_type: str, content: str, phase_cn: str, auto_hide: bool) -> None:
        self.auto_running = False
        self.report_popup_visible = True
        self.report_popup_type = report_type
        self.report_popup_title = f"{phase_cn}战术分析报告"
        self.report_popup_content = content
        self.report_popup_auto_close_at = time.time() + self.REPORT_POPUP_SHOW_SECONDS if auto_hide else None
        self.report_popup_ok_hold_start = None
        self.report_popup_wait_release = False

    def _close_report_popup(self) -> None:
        self.report_popup_visible = False
        self.report_popup_type = None
        self.report_popup_title = ""
        self.report_popup_content = ""
        self.report_popup_auto_close_at = None
        self.report_popup_ok_hold_start = None
        self.report_popup_wait_release = False

    def _maybe_trigger_phase_reports(self, before_seconds: int, after_seconds: int) -> None:
        if (
            (not self.halftime_report_generated)
            and before_seconds < self.HALF_TIME_SECONDS <= after_seconds
            and after_seconds < self.MATCH_END_SECONDS
        ):
            self._request_phase_report("halftime", "中场", auto_open=True)
        if (not self.fulltime_report_generated) and before_seconds < self.MATCH_END_SECONDS <= after_seconds:
            self._request_phase_report("fulltime", "终场", auto_open=True)

    def _update_report_popup_timeout(self) -> None:
        if (
            self.report_popup_visible
            and self.report_popup_auto_close_at is not None
            and time.time() >= self.report_popup_auto_close_at
        ):
            self._close_report_popup()

    def _draw_report_popup(self) -> None:
        if not self.report_popup_visible:
            return
        sw, sh = self.screen.get_size()
        mask = pygame.Surface((sw, sh), pygame.SRCALPHA)
        mask.fill((0, 0, 0, 160))
        self.screen.blit(mask, (0, 0))

        pw = min(int(sw * 0.72), 980)
        ph = min(int(sh * 0.74), 620)
        popup = pygame.Rect((sw - pw) // 2, (sh - ph) // 2, pw, ph)
        pygame.draw.rect(self.screen, (26, 26, 26), popup, border_radius=10)
        pygame.draw.rect(self.screen, self.PANEL_BORDER, popup, 1, border_radius=10)

        x = popup.x + 14
        y = popup.y + 12
        self._draw_text(self.report_popup_title, x, y, self.PANEL_ACCENT, self.font_title)
        y += 30
        hint = "报告展示中：自动关闭30秒，或 OK手势保持1秒关闭"
        self._draw_text(hint, x, y, self.PANEL_MUTED, self.font_small)
        y += 24

        content_rect = pygame.Rect(x, y, popup.width - 28, popup.height - (y - popup.y) - 14)
        pygame.draw.rect(self.screen, (18, 18, 18), content_rect, border_radius=6)
        pygame.draw.rect(self.screen, self.PANEL_BORDER, content_rect, 1, border_radius=6)

        tx = content_rect.x + 8
        ty = content_rect.y + 8
        line_h = 18
        max_lines = max(1, (content_rect.height - 12) // line_h)
        max_w = content_rect.width - 16
        lines: List[str] = []
        for para in self.report_popup_content.splitlines() or [""]:
            if not para:
                lines.append("")
                continue
            cur = ""
            for ch in para:
                cand = cur + ch
                if self.font_small.size(cand)[0] <= max_w:
                    cur = cand
                else:
                    lines.append(cur)
                    cur = ch
            if cur:
                lines.append(cur)
        lines = lines[:max_lines]
        for ln in lines:
            self._draw_text(ln, tx, ty, self.PANEL_TEXT, self.font_small)
            ty += line_h

    def _handle_report_popup_ok_close(self) -> None:
        if not self.report_popup_visible:
            return
        if self.report_popup_wait_release:
            if not (self.hand_detected and self.hand_keypoints and self.hand_keypoints.get("is_ok_gesture", False)):
                self.report_popup_wait_release = False
            return
        if not (self.hand_detected and self.hand_keypoints and self.hand_keypoints.get("is_ok_gesture", False)):
            self.report_popup_ok_hold_start = None
            return
        now = time.time()
        if self.report_popup_ok_hold_start is None:
            self.report_popup_ok_hold_start = now
            return
        if now - self.report_popup_ok_hold_start >= self.REPORT_OK_CLOSE_SECONDS:
            self._close_report_popup()
            self.report_popup_wait_release = True
            self.report_popup_ok_hold_start = None

    def _handle_auto_tick(self) -> None:
        if not self.auto_running:
            return
        now = time.time()
        while now - self.last_auto_tick_time >= self.auto_slot_seconds:
            self.last_auto_tick_time += self.auto_slot_seconds
            if self.engine.state.match_seconds >= self.MATCH_END_SECONDS:
                self.auto_running = False
                break
            self._manual_step()
            if not self.auto_running:
                break

    def _draw_text(self, text: str, x: int, y: int, color=None, font=None, target: Optional[pygame.Surface] = None) -> int:
        surf = (font or self.font_small).render(text, True, color or self.PANEL_TEXT)
        (target or self.screen).blit(surf, (x, y))
        return y + surf.get_height() + 4

    def _set_camera_frame(self, frame_bgr) -> None:
        try:
            h, w = frame_bgr.shape[:2]
            max_w = 640
            if w > max_w:
                scale = max_w / float(w)
                frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
            frame_rgb = frame_bgr[:, :, ::-1]
            self.camera_surface_raw = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            self.camera_available = True
            self.last_good_frame_time = time.time()
            self.camera_offline_reported = False
        except Exception:
            self.camera_available = False

    def _open_camera_if_needed(self) -> bool:
        if self._camera_cap is not None:
            return True
        now = time.time()
        if now - self.last_camera_reinit_attempt < self.camera_reinit_interval:
            return False
        self.last_camera_reinit_attempt = now
        for idx in self.camera_indices:
            cap_test = None
            try:
                cap_test = cv2.VideoCapture(idx)
                if not cap_test.isOpened():
                    if cap_test is not None:
                        cap_test.release()
                    continue
                ret, _frame = cap_test.read()
                if not ret:
                    cap_test.release()
                    continue
                self._camera_cap = cap_test
                self.camera_fail_streak = 0
                self._append_log(f"[摄像头] 重连成功（索引{idx}）")
                self.camera_offline_reported = False
                return True
            except Exception:
                if cap_test is not None:
                    cap_test.release()
                continue
        return False

    def _camera_worker_loop(self) -> None:
        while not self._camera_stop_event.is_set():
            if not self._open_camera_if_needed():
                time.sleep(0.05)
                continue

            try:
                ret, frame = self._camera_cap.read() if self._camera_cap is not None else (False, None)
            except Exception:
                ret, frame = False, None

            if ret and frame is not None:
                self.camera_fail_streak = 0
                ts = time.time()
                with self._camera_lock:
                    self._latest_camera_frame = frame
                    self._latest_camera_ts = ts
                continue

            self.camera_fail_streak += 1
            if self.camera_fail_streak >= self.camera_fail_limit:
                if self._camera_cap is not None:
                    try:
                        self._camera_cap.release()
                    except Exception:
                        pass
                    self._camera_cap = None
                self.camera_fail_streak = 0
                if not self.camera_offline_reported:
                    self._append_log("[摄像头] 暂不可用，已进入自动重连")
                    self.camera_offline_reported = True
            time.sleep(0.01)

    def _start_camera_thread(self) -> None:
        if self._camera_thread is not None and self._camera_thread.is_alive():
            return
        self._camera_stop_event.clear()
        self._camera_thread = threading.Thread(target=self._camera_worker_loop, name="camera-capture-thread", daemon=True)
        self._camera_thread.start()

    def _stop_camera_thread(self) -> None:
        self._camera_stop_event.set()
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=1.0)
        self._camera_thread = None
        if self._camera_cap is not None:
            try:
                self._camera_cap.release()
            except Exception:
                pass
            self._camera_cap = None

    def _get_latest_camera_frame(self, max_age: float = 1.5):
        with self._camera_lock:
            frame = self._latest_camera_frame
            ts = self._latest_camera_ts
        if frame is None:
            return None
        if time.time() - ts > max_age:
            return None
        return frame.copy()

    def _blit_fit_aspect(self, src: pygame.Surface, target_rect: pygame.Rect) -> None:
        sw, sh = src.get_size()
        tw, th = target_rect.width, target_rect.height
        if sw <= 0 or sh <= 0 or tw <= 0 or th <= 0:
            return
        scale = min(tw / sw, th / sh)
        nw = max(1, int(sw * scale))
        nh = max(1, int(sh * scale))
        img = pygame.transform.smoothscale(src, (nw, nh))
        dx = target_rect.x + (tw - nw) // 2
        dy = target_rect.y + (th - nh) // 2
        self.screen.blit(img, (dx, dy))

    def _draw_compact_strategy_row(
        self,
        title: str,
        group_key: str,
        options: List[str],
        selected: str,
        x: int,
        y: int,
        w: int,
        enabled: bool,
    ) -> int:
        row_h = 24
        title_w = max(82, int(w * 0.28))
        self._draw_text(title, x, y + 2, self.PANEL_TEXT)
        option_x = x + title_w
        option_w = max(64, (w - title_w - 8) // max(1, len(options)))

        def display_name(raw: str) -> str:
            return "平衡" if raw == "默认平衡" else raw

        for i, op in enumerate(options):
            r = pygame.Rect(option_x + i * option_w, y, option_w - 6, row_h)
            active = op == selected
            disabled = (not enabled) and (not active)
            bg = (54, 80, 100) if active else self.PANEL_CARD
            if disabled:
                bg = (44, 44, 44)
            pygame.draw.rect(self.screen, bg, r, border_radius=6)
            pygame.draw.rect(self.screen, self.PANEL_BORDER, r, 1, border_radius=6)
            cx = r.x + 10
            cy = r.y + r.height // 2
            pygame.draw.circle(self.screen, (200, 200, 200) if not disabled else (110, 110, 110), (cx, cy), 6, 2)
            if active:
                pygame.draw.circle(self.screen, self.PANEL_ACCENT, (cx, cy), 3)
            txt_col = self.PANEL_ACCENT if active else (self.PANEL_MUTED if disabled else self.PANEL_TEXT)
            self._draw_text(display_name(op), r.x + 20, r.y + 2, txt_col)
            self.click_regions.append({"rect": r, "kind": "strategy", "value": f"{group_key}:{op}", "enabled": not disabled})
        return y + row_h + 6

    @staticmethod
    def _strategy_short_name(group_key: str, raw: str) -> str:
        if raw == "默认平衡":
            return "平衡"
        mapping = {
            "height": {
                "前压": "前压",
                "后撤": "后撤",
                "深度防守": "深防",
            },
            "tempo": {
                "控球推进": "控推",
                "快速反击": "快反",
            },
            "channel": {
                "边路倾斜": "边路",
                "中路渗透": "中路",
            },
        }
        return mapping.get(group_key, {}).get(raw, raw[:4])

    def _draw_mid_strategy_panel(self, pitch_rect: pygame.Rect, image_rect: pygame.Rect) -> None:
        # 将策略区放到球场图右侧空白条（A上可编辑，B下只读）
        target_w = max(140, min(220, int(pitch_rect.width * 0.16)))
        strip_w = max(96, int(target_w * 0.60))
        strip_x = pitch_rect.right - strip_w - 8
        # 若右侧空白不足，允许轻微覆盖到球场图右缘，确保策略栏始终可见
        min_x = image_rect.left + 8
        if strip_x < min_x:
            strip_x = min_x
        strip_h = max(160, image_rect.height - 56)
        strip = pygame.Rect(strip_x, image_rect.top + 44, strip_w, strip_h)
        pygame.draw.rect(self.screen, (14, 16, 20), strip, border_radius=8)
        pygame.draw.rect(self.screen, (230, 90, 90), strip, 2, border_radius=8)

        snap = self.engine.get_snapshot()
        qa = snap["A"]["quota"]
        strat_enabled = qa.strategy_left > 0
        tb = self.engine.team_b
        now = time.time()
        select_progress = 0.0
        select_target_value = None
        if (
            self.strategy_hold_start_time is not None
            and self.strategy_hold_target_value
            and self.is_grabbing
        ):
            select_target_value = self.strategy_hold_target_value
            select_progress = min(1.0, (now - self.strategy_hold_start_time) / self.STRATEGY_SELECT_HOLD_SECONDS)
        ok_progress = 0.0
        if self.strategy_confirm_pending and self.ok_hold_start_time is not None:
            ok_progress = min(1.0, (now - self.ok_hold_start_time) / self.STRATEGY_CONFIRM_OK_SECONDS)

        x = strip.x + 6
        y = strip.y + 8
        w = strip.width - 12
        button_h = 22
        gap = 4

        def draw_group(group_title: str, group_key: str, options: List[str], selected: str, editable: bool) -> int:
            nonlocal y
            y = self._draw_text(group_title, x, y, self.PANEL_TEXT)
            for op in options:
                r = pygame.Rect(x, y, w, button_h)
                item_value = f"{group_key}:{op}"
                active = op == selected
                disabled = (not editable) or ((not strat_enabled) and (not active))
                bg = (56, 88, 118) if active else (36, 36, 36)
                if disabled and not active:
                    bg = (30, 30, 30)
                pygame.draw.rect(self.screen, bg, r, border_radius=6)
                pygame.draw.rect(self.screen, self.PANEL_BORDER, r, 1, border_radius=6)
                if editable and item_value == select_target_value and select_progress > 0:
                    pw = int((r.width - 4) * select_progress)
                    if pw > 0:
                        pbar = pygame.Rect(r.x + 2, r.bottom - 4, pw, 2)
                        pygame.draw.rect(self.screen, self.PANEL_ACCENT, pbar, border_radius=1)
                name = self._strategy_short_name(group_key, op)
                color = self.PANEL_TEXT if not disabled else self.PANEL_MUTED
                self._draw_text(name, r.x + 8, r.y + 2, color)
                if editable:
                    self.click_regions.append(
                        {"rect": r, "kind": "strategy", "value": item_value, "enabled": not disabled}
                    )
                y += button_h + gap
            y += 2
            return y

        self._draw_text("A队策略", x, y, self.PANEL_ACCENT, self.font_small)
        y += 22
        if self.strategy_confirm_pending:
            y = self._draw_text("OK确认中（1秒）", x, y, self.PANEL_MUTED)
            track = pygame.Rect(x, y + 1, w, 6)
            pygame.draw.rect(self.screen, (44, 44, 44), track, border_radius=3)
            fill = pygame.Rect(track.x, track.y, int(track.width * ok_progress), track.height)
            if fill.width > 0:
                pygame.draw.rect(self.screen, self.GREEN, fill, border_radius=3)
            y += 12
        draw_group("高度组", "height", self.height_options, self.a_strategy_select["height"], editable=True)
        draw_group("节奏组", "tempo", self.tempo_options, self.a_strategy_select["tempo"], editable=True)
        draw_group("通道组", "channel", self.channel_options, self.a_strategy_select["channel"], editable=True)

        if not strat_enabled:
            y = self._draw_text("A策略配额不足", x, y, self.RED)
            y += 2

        y += 4
        self._draw_text("B队策略", x, y, self.PANEL_ACCENT, self.font_small)
        y += 22
        y = self._draw_text(f"高度: {self._strategy_short_name('height', tb.strategy_height)}", x, y, self.PANEL_MUTED)
        y = self._draw_text(f"节奏: {self._strategy_short_name('tempo', tb.strategy_tempo)}", x, y, self.PANEL_MUTED)
        self._draw_text(f"通道: {self._strategy_short_name('channel', tb.strategy_channel)}", x, y, self.PANEL_MUTED)

    def _draw_feature_bars_compact(self, x: int, y: int, w: int) -> int:
        formation = self.preview_formation_engine or self.engine.team_a.formation
        base, weighted, delta, _risk, _route = self.engine.calc_features_for_config(
            formation,
            self.a_strategy_select["height"],
            self.a_strategy_select["tempo"],
            self.a_strategy_select["channel"],
        )
        y = self._draw_text("A队8维评分（基础 vs 加权）", x, y, self.PANEL_TEXT, self.font_body)
        col_w = (w - 8) // 2
        bar_h = 8
        row_h = 34
        for idx, k in enumerate(FEATURE_KEYS):
            col = idx % 2
            row = idx // 2
            bx = x + col * (col_w + 8)
            by = y + row * row_h
            self._draw_text(FEATURE_LABELS.get(k, k), bx, by, self.PANEL_MUTED)
            bg = pygame.Rect(bx, by + 16, col_w, bar_h)
            pygame.draw.rect(self.screen, (48, 48, 48), bg, border_radius=3)
            # 阵型基础评分（蓝色）
            bw = int(col_w * max(0.0, min(1.0, base[k])))
            if bw > 0:
                pygame.draw.rect(self.screen, (90, 150, 230), (bx, by + 16, bw, bar_h), border_radius=3)
            # 策略加权变化：正向右(绿)，负向左(红)
            d = delta[k]
            if abs(d) > 1e-6:
                dw = int(min(col_w, abs(d) * col_w))
                anchor = bx + bw
                if d > 0:
                    x0 = min(bx + col_w, anchor)
                    dw2 = min(dw, bx + col_w - x0)
                    if dw2 > 0:
                        pygame.draw.rect(self.screen, self.GREEN, (x0, by + 26, dw2, 3), border_radius=2)
                else:
                    x1 = max(bx, anchor - dw)
                    dw2 = max(0, anchor - x1)
                    if dw2 > 0:
                        pygame.draw.rect(self.screen, self.RED, (x1, by + 26, dw2, 3), border_radius=2)
        return y + 4 * row_h + 4

    def _draw_event_log(self, x: int, y: int, w: int, h: int) -> int:
        card = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, self.PANEL_CARD, card, border_radius=8)
        pygame.draw.rect(self.screen, self.PANEL_BORDER, card, 1, border_radius=8)
        self._draw_text("比赛日志（最近）", x + 8, y + 6, self.PANEL_TEXT, self.font_body)
        yy = y + 34
        line_h = 18
        max_lines = max(1, (h - 42) // line_h)
        max_text_w = max(40, w - 16)

        def wrap_line(text: str) -> List[str]:
            if not text:
                return [""]
            out: List[str] = []
            cur = ""
            for ch in text:
                cand = cur + ch
                if self.font_small.size(cand)[0] <= max_text_w:
                    cur = cand
                    continue
                if cur:
                    out.append(cur)
                    cur = ch
                else:
                    out.append(ch)
                    cur = ""
            if cur:
                out.append(cur)
            return out or [text]

        wrapped_lines: List[str] = []
        for ln in self.log_lines:
            wrapped_lines.extend(wrap_line(ln))
        lines = wrapped_lines[-max_lines:]
        for ln in lines:
            if yy + line_h > y + h - 4:
                break
            self._draw_text(ln, x + 8, yy, self.PANEL_MUTED)
            yy += line_h
        return y + h + 8

    def _draw_pitch_overlay_info(self, image_rect: pygame.Rect) -> None:
        snap = self.engine.get_snapshot()
        score_a, score_b = snap["score"]
        gesture_text = "抓取中" if self.is_grabbing else ("未检测到手" if not self.hand_detected else "悬停")

        # 顶部中间：三行信息（时间、比分、手势）
        big_font = gfi.get_font(24, bold=True)
        mid_font = gfi.get_font(20, bold=True)
        small_font = gfi.get_font(16)
        center_x = image_rect.centerx
        top_y = image_rect.top + 4
        time_text = f"{format_match_time(snap['match_seconds'])} / 90:00"
        score_text = f"A队 {score_a}:{score_b} B队"
        hand_text = f"手势：{gesture_text}"

        ts = big_font.render(time_text, True, (245, 245, 245))
        ss = mid_font.render(score_text, True, (245, 245, 245))
        hs = small_font.render(hand_text, True, (170, 170, 170))
        self.screen.blit(ts, ts.get_rect(center=(center_x, top_y + 10)))
        self.screen.blit(ss, ss.get_rect(center=(center_x, top_y + 36)))
        self.screen.blit(hs, hs.get_rect(center=(center_x, top_y + 58)))

        def draw_team_block(team: str, is_left: bool) -> None:
            tactics = self.engine.team_a if team == "A" else self.engine.team_b
            z = self.team_zone_view.get(team, snap["zone"])
            route = self.team_route_view.get(team, "中路")
            events = self.team_event_traces.get(team, [])[-5:]
            bw = 360
            x = image_rect.left + 8 if is_left else image_rect.right - bw - 8
            y = image_rect.top + 4
            # 透明背景：仅绘制文字（加细描边阴影增强可读性）
            title = f"{TEAM_CN.get(team, team)} 阵型:{tactics.formation}"
            self._draw_text(title, x + 6, y + 2, self.PANEL_TEXT)
            self._draw_text(
                f"区域:{zone_tuple_to_cn(z)} | 线路:{route}",
                x + 6,
                y + 22,
                self.PANEL_MUTED,
            )
            if not events:
                self._draw_text("暂无事件", x + 6, y + 40, self.PANEL_MUTED)
            else:
                yy = y + 40
                for line in events:
                    self._draw_text(line if len(line) <= 34 else (line[:34] + "..."), x + 6, yy, self.PANEL_MUTED)
                    yy += 16

        draw_team_block("A", is_left=True)
        draw_team_block("B", is_left=False)

    def _draw_pitch_bottom_controls(self, pitch_rect: pygame.Rect, image_rect: pygame.Rect) -> None:
        # 左侧球场图下方操作按钮：固定放在球场区域下方独立条带
        controls = [
            {"id": "start", "label": "开始"},
            {"id": "pause", "label": "暂停"},
            {"id": "reset", "label": "重置"},
            {"id": "strategy", "label": "半场报告"},
            {"id": "report", "label": "全场报告"},
        ]
        self.pitch_control_regions = []
        self.pitch_control_bar_rect = None
        if not controls:
            return

        gap = 8
        btn_h = 36
        left = image_rect.left
        avail_w = image_rect.width
        per_row = len(controls)
        rows = 1
        total_h = btn_h
        top = pitch_rect.bottom + 20
        max_bottom = self.screen.get_height() - 44
        if top + total_h > max_bottom:
            top = max(10, max_bottom - total_h)
        self.pitch_control_bar_rect = pygame.Rect(left, top, avail_w, total_h)

        progress = 0.0
        if self.pitch_control_hold_target and self.pitch_control_hold_start_time is not None:
            progress = min(1.0, (time.time() - self.pitch_control_hold_start_time) / self.pitch_control_hold_seconds)

        idx = 0
        for row in range(rows):
            row_items = controls[idx: idx + per_row]
            if not row_items:
                break
            row_w = avail_w
            btn_w = (row_w - (len(row_items) - 1) * gap) // len(row_items)
            y = top + row * (btn_h + gap)
            for col, item in enumerate(row_items):
                x = left + col * (btn_w + gap)
                rect = pygame.Rect(x, y, btn_w, btn_h)
                self.pitch_control_regions.append({"id": item["id"], "rect": rect})

                is_hover = self.hand_detected and rect.collidepoint((self.hand_x, self.hand_y))
                is_holding = (
                    is_hover
                    and self.is_grabbing
                    and self.pitch_control_hold_target == item["id"]
                    and self.pitch_control_hold_start_time is not None
                )

                bg = (48, 48, 48)
                if item["id"] in {"strategy", "report"}:
                    bg = (42, 42, 42)
                if is_hover:
                    bg = (58, 72, 86)
                pygame.draw.rect(self.screen, bg, rect, border_radius=8)
                pygame.draw.rect(self.screen, self.PANEL_BORDER, rect, 1, border_radius=8)

                if is_holding:
                    fill_w = int(rect.width * progress)
                    if fill_w > 0:
                        fill_rect = pygame.Rect(rect.x, rect.y, fill_w, rect.height)
                        pygame.draw.rect(self.screen, (78, 122, 96), fill_rect, border_radius=8)
                        pygame.draw.rect(self.screen, self.PANEL_BORDER, rect, 1, border_radius=8)

                text_color = self.PANEL_TEXT if item["id"] in {"start", "pause", "reset"} else self.PANEL_MUTED
                label = self.font_body.render(item["label"], True, text_color)
                self.screen.blit(label, label.get_rect(center=rect.center))

            idx += len(row_items)

    def _trigger_pitch_control_action(self, action_id: str) -> None:
        if action_id == "start":
            if self.engine.state.match_seconds >= self.MATCH_END_SECONDS:
                self._append_structured_log("比赛已结束，无法开始自动推进")
                return
            if not self.auto_running:
                self.auto_running = True
                self.last_auto_tick_time = time.time()
                self._append_structured_log("自动推进启动")
            return
        if action_id == "pause":
            if self.auto_running:
                self.auto_running = False
                self._append_structured_log("自动推进暂停")
            return
        if action_id == "reset":
            self._reset_match_state_only()
            return
        if action_id == "strategy":
            if self.report_cache.get("halftime"):
                self._open_report_popup("halftime", self.report_cache["halftime"], "中场", auto_hide=True)
            elif self.report_generating.get("halftime", False):
                self._open_report_popup("halftime", "正在生成中场战术分析报告，请稍候...", "中场", auto_hide=False)
            else:
                self._append_structured_log("中场报告尚未生成（需到45:00后）")
            return
        if action_id == "report":
            if self.report_cache.get("fulltime"):
                self._open_report_popup("fulltime", self.report_cache["fulltime"], "终场", auto_hide=True)
            elif self.report_generating.get("fulltime", False):
                self._open_report_popup("fulltime", "正在生成终场战术分析报告，请稍候...", "终场", auto_hide=False)
            else:
                self._append_structured_log("终场报告尚未生成（需到90:00后）")
            return

    def _handle_pitch_control_gesture(self) -> None:
        if not self.pitch_control_regions:
            self.pitch_control_hold_start_time = None
            self.pitch_control_hold_target = None
            return
        if self.pitch_control_wait_release:
            if not self.is_grabbing:
                self.pitch_control_wait_release = False
            return
        if not self.hand_detected:
            self.pitch_control_hold_start_time = None
            self.pitch_control_hold_target = None
            return

        hovered_id = None
        hand_pos = (self.hand_x, self.hand_y)
        hover_padding = 10
        for item in self.pitch_control_regions:
            if item["rect"].inflate(hover_padding * 2, hover_padding * 2).collidepoint(hand_pos):
                hovered_id = str(item["id"])
                break

        if not hovered_id or not self.is_grabbing:
            self.pitch_control_hold_start_time = None
            self.pitch_control_hold_target = None
            return

        now = time.time()
        if self.pitch_control_hold_target != hovered_id:
            self.pitch_control_hold_target = hovered_id
            self.pitch_control_hold_start_time = now
            return
        if self.pitch_control_hold_start_time is None:
            self.pitch_control_hold_start_time = now
            return

        if now - self.pitch_control_hold_start_time >= self.pitch_control_hold_seconds:
            self._trigger_pitch_control_action(hovered_id)
            self.pitch_control_wait_release = True
            self.pitch_control_hold_start_time = None
            self.pitch_control_hold_target = None

    def _draw_right_panel(self, pitch_rect: pygame.Rect) -> None:
        sw, sh = self.screen.get_size()
        panel = pygame.Rect(pitch_rect.right + 10, 10, sw - pitch_rect.right - 20, sh - 20)
        pygame.draw.rect(self.screen, self.PANEL_BG, panel, border_radius=8)
        pygame.draw.rect(self.screen, self.PANEL_BORDER, panel, 1, border_radius=8)

        self.scroll_viewport = None
        x = panel.x + 10
        y = panel.y + 8
        w = panel.width - 20

        snap = self.engine.get_snapshot()
        y += 2

        # 固定摄像头区域（同屏展示，不翻页）
        cam_h = max(180, int(panel.height * 0.30))
        cam_card = pygame.Rect(x, y, w, cam_h)
        pygame.draw.rect(self.screen, self.PANEL_CARD, cam_card, border_radius=8)
        pygame.draw.rect(self.screen, self.PANEL_BORDER, cam_card, 1, border_radius=8)
        self._draw_text("摄像头与手部骨架", cam_card.x + 8, cam_card.y + 6, self.PANEL_TEXT, target=self.screen)
        cam_inner = cam_card.inflate(-12, -30)
        cam_inner.y += 18
        if self.camera_available and self.camera_surface_raw is not None:
            pygame.draw.rect(self.screen, (18, 18, 18), cam_inner, border_radius=6)
            self._blit_fit_aspect(self.camera_surface_raw, cam_inner)
        else:
            pygame.draw.rect(self.screen, (18, 18, 18), cam_inner, border_radius=6)
            self._draw_text("未检测到摄像头画面", cam_inner.x + 10, cam_inner.y + 10, self.PANEL_MUTED, target=self.screen)
        y = cam_card.bottom + 8

        qa = snap["A"]["quota"]
        qb = snap["B"]["quota"]
        y = self._draw_text(
            f"调整配额 A: 阵型{qa.formation_left}/1 策略{qa.strategy_left}/2 | "
            f"B: 阵型{qb.formation_left}/1 策略{qb.strategy_left}/2",
            x,
            y,
            self.PANEL_MUTED,
        )
        y += 2

        route = snap["A"]["route"]
        y = self._draw_text(
            f"A队路线权重 左/中/右 = {route[0]:.2f}/{route[1]:.2f}/{route[2]:.2f}",
            x,
            y,
            self.PANEL_MUTED,
        )
        y += 2

        a_stats = snap["stats"]["A"]
        b_stats = snap["stats"]["B"]
        a_sh = max(1, a_stats.get("shots", 0))
        b_sh = max(1, b_stats.get("shots", 0))
        a_sr = a_stats.get("shots_on_target", 0) / a_sh
        b_sr = b_stats.get("shots_on_target", 0) / b_sh
        y = self._draw_text(
            f"专业统计 A: 射门{a_stats['shots']} 射正{a_stats['shots_on_target']} 射正率{a_sr:.0%} 失误{a_stats['turnovers']} 反击{a_stats['counterattacks']}",
            x,
            y,
            self.PANEL_MUTED,
        )
        y = self._draw_text(
            f"B: 射门{b_stats['shots']} 射正{b_stats['shots_on_target']} 射正率{b_sr:.0%} 失误{b_stats['turnovers']} 反击{b_stats['counterattacks']}",
            x,
            y,
            self.PANEL_MUTED,
        )
        y += 2

        log_h = max(100, panel.bottom - y - 8)
        self._draw_event_log(x, y, w, log_h)

    def _on_panel_click(self, pos: Tuple[int, int]) -> None:
        for item in self.click_regions:
            rect = item["rect"]
            if not rect.collidepoint(pos):
                continue
            if not item.get("enabled", True):
                self._append_structured_log("A队调整失败：当前选项已禁用（配额不足）")
                return
            kind = str(item["kind"])
            value = str(item["value"])
            if kind == "strategy":
                group, opt = value.split(":", 1)
                old = self.a_strategy_select[group]
                self.a_strategy_select[group] = opt
                before_state = (
                    self.engine.team_a.strategy_height,
                    self.engine.team_a.strategy_tempo,
                    self.engine.team_a.strategy_channel,
                )
                self._sync_a_to_engine(force=False)
                after_state = (
                    self.engine.team_a.strategy_height,
                    self.engine.team_a.strategy_tempo,
                    self.engine.team_a.strategy_channel,
                )
                if (
                    before_state == after_state
                    and old != opt
                ):
                    self._append_structured_log("A队策略调整未生效（配额不足）")
                elif old != opt:
                    self._append_structured_log(f"A队策略调整 {group}:{old}->{opt}")
                return

    def maybe_trigger_team_b_auto_switch(self) -> None:
        # 覆盖基类：融合版中由状态机时间窗计划驱动，不使用“每10秒自动换阵”
        return

    def handle_ok_gesture_confirmation(self) -> None:
        """
        覆盖基类：
        - 仅确认A队阵型，不再同步强制改B队
        - 确认成功后写入状态机
        """
        if self.report_popup_visible:
            self._handle_report_popup_ok_close()
            return
        if self.strategy_confirm_pending:
            self._confirm_pending_strategy_with_ok()
            return
        if self.hand_keypoints and self.hand_keypoints.get("is_ok_gesture", False) and self.hand_detected:
            current_time = time.time()
            if self.ok_confirmed_wait_release:
                return
            if self.ok_hold_start_time is None:
                self.ok_hold_start_time = current_time
                self.ok_hold_elapsed = 0.0
                return

            self.ok_hold_elapsed = current_time - self.ok_hold_start_time
            if (
                self.ok_hold_elapsed >= self.ok_hold_duration
                and current_time - self.last_ok_gesture_time > self.ok_gesture_cooldown
            ):
                recognized = self.recognize_formation()
                target = self.find_closest_standard_formation(recognized)
                if target:
                    prev = self.selected_formation_a
                    if prev == target and not self.a_position_dirty_pending_ok:
                        self.confirmation_message = f"无需调整（已是 {target}）"
                        self.confirmation_timer = 180
                    else:
                        self.last_adjustment_from = prev
                        self.last_adjustment_to = target
                        formation_ok = self._apply_a_formation_only_to_engine(target)
                        if not formation_ok:
                            self.confirmation_message = f"A队确认失败：{target}（阵型配额不足）"
                            self.confirmation_timer = 200
                        else:
                            self.confirmation_message = f"A队确认: {target}"
                            self.confirmation_timer = 200
                            success = self.adjust_formation_to_standard(target, team_name="A")
                            if success:
                                self.selected_formation_a = target
                                self.a_position_dirty_pending_ok = False
                                self._append_structured_log(f"A队手势确认阵型 -> {self._ui_to_engine(target)}")
                else:
                    self.confirmation_message = "没有匹配阵型"
                    self.confirmation_timer = 120

                self.last_ok_gesture_time = current_time
                self.ok_confirmed_wait_release = True
                self.ok_hold_start_time = None
                self.ok_hold_elapsed = 0.0
        else:
            self.ok_hold_start_time = None
            self.ok_hold_elapsed = 0.0
            self.ok_confirmed_wait_release = False

    def draw_status_info(self) -> None:
        sw, sh = self.screen.get_size()
        hint = "空格:开始/暂停自动推进  N:手动推进15秒  M:快进到中场/终场  R:重置比赛"
        self.screen.blit(self.font_small.render(hint, True, (150, 150, 150)), (20, sh - 30))

        # 红框：A队阵型调整过程中的识别/匹配结果
        adjusting = (self.dragging_player is not None and self.dragging_player.team == "A") or self.a_position_dirty_pending_ok
        if adjusting:
            box_w = 460
            box_h = 46
            if self.last_pitch_image_rect is not None:
                base_y = self.last_pitch_image_rect.bottom - box_h - 10
            else:
                base_y = sh - box_h - 16
            box = pygame.Rect((sw - box_w) // 2, base_y, box_w, box_h)
            if self.pitch_control_bar_rect is not None and box.colliderect(self.pitch_control_bar_rect):
                box.y = max(12, self.pitch_control_bar_rect.y - box_h - 8)
            pygame.draw.rect(self.screen, (60, 24, 24), box, border_radius=6)
            pygame.draw.rect(self.screen, (230, 70, 70), box, 2, border_radius=6)
            text = f"A队识别:{self.live_recognized_formation} | 匹配:{self.live_matched_formation}"
            self._draw_text(text, box.x + 10, box.y + 12, (255, 220, 140))

        if self.confirmation_timer > 0 and self.confirmation_message:
            msg = self.font_body.render(self.confirmation_message, True, (255, 200, 120))
            rect = msg.get_rect(center=(sw // 2, sh - 92))
            self.screen.blit(msg, rect)

    def _reset_match_state_only(self) -> None:
        self.engine.reset()
        self.b_window_plan.clear()
        self._ensure_b_window_plan(0)
        self.last_report = None
        self.halftime_paused = False
        self.auto_running = False
        self.last_auto_tick_time = time.time()
        self.log_lines.clear()
        self.team_event_traces = {"A": [], "B": []}
        self.team_zone_view = {"A": ("MC", "C"), "B": ("MC", "C")}
        self.team_route_view = {"A": "中路", "B": "中路"}
        self.preview_formation_engine = self.engine.team_a.formation
        self.halftime_report_generated = False
        self.fulltime_report_generated = False
        self.report_cache.clear()
        self.report_generating = {"halftime": False, "fulltime": False}
        self.report_auto_open_pending = {"halftime": False, "fulltime": False}
        with self.report_result_lock:
            self.report_result_queue.clear()
        self._close_report_popup()
        self._sync_a_to_engine(force=True)
        self._sync_b_to_engine(force=True)
        self._append_structured_log("比赛状态已重置（球场与手势状态保留）")

    def _fast_forward_to_next_phase(self) -> None:
        cur = self.engine.state.match_seconds
        if cur >= self.MATCH_END_SECONDS:
            self._append_structured_log("比赛已结束，无法继续快进")
            return

        if cur < self.HALF_TIME_SECONDS:
            target = self.HALF_TIME_SECONDS
            phase_text = "中场"
        else:
            target = self.MATCH_END_SECONDS
            phase_text = "终场"

        self.auto_running = False
        self._append_structured_log(f"开始快速模拟到{phase_text}（M键）")
        while self.engine.state.match_seconds < target:
            self._manual_step()
            if self.report_popup_visible:
                # 阶段报告弹窗出现后停止快进，等待用户查看
                break
        self._append_structured_log(f"快速模拟结束：当前时间 {format_match_time(self.engine.state.match_seconds)}")

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.VIDEORESIZE:
                self.create_players()
            elif event.type == pygame.KEYDOWN:
                if self.report_popup_visible:
                    continue
                if event.key == pygame.K_SPACE:
                    if self.engine.state.match_seconds >= self.MATCH_END_SECONDS:
                        continue
                    self.auto_running = not self.auto_running
                    self.last_auto_tick_time = time.time()
                    self._append_structured_log(f"自动推进{'启动' if self.auto_running else '暂停'}")
                elif event.key == pygame.K_n:
                    if self.engine.state.match_seconds < self.MATCH_END_SECONDS:
                        self._manual_step()
                elif event.key == pygame.K_m:
                    self._fast_forward_to_next_phase()
                elif event.key == pygame.K_r:
                    self._reset_match_state_only()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self._on_panel_click(event.pos)
        return True

    def run(self) -> None:
        running = True
        while running:
            running = self.handle_events()
            if not running:
                break

            self._flush_report_results()
            self._update_report_popup_timeout()
            self._handle_auto_tick()
            self.screen.fill(gfi.BACKGROUND_GRAY)

            frame = self._get_latest_camera_frame(max_age=1.5)
            frame_available = frame is not None

            if frame_available and frame is not None:
                now = time.time()
                # 手势推理降频，降低CPU占用并缓解卡顿
                if now - self.last_gesture_infer_time >= self.gesture_interval_seconds:
                    self.last_gesture_infer_time = now
                    self.is_grabbing, hand_coords, self.hand_keypoints = self.detect_gesture(frame)
                    self.hand_detected = hand_coords is not None
                    if hand_coords:
                        mapped = self.map_hand_to_screen(hand_coords)
                        if mapped:
                            self._update_hand_cursor(mapped)
                self.handle_ok_gesture_confirmation()
                if not self.report_popup_visible:
                    self.handle_gesture_interaction()
                self.update_smooth_movement()
                if self.hand_keypoints:
                    frame = self.draw_hand_skeleton(frame, self.hand_keypoints)
                self._set_camera_frame(frame)
            else:
                # 短时掉帧时保留最近画面与状态，避免闪断感
                if time.time() - self.last_good_frame_time > 1.5:
                    self.hand_detected = False
                    self.is_grabbing = False
                    self.hand_keypoints = None
                    self.camera_available = False
                    self.cursor_filtered_x = None
                    self.cursor_filtered_y = None
                self.update_smooth_movement()

            self._refresh_live_formation_match()
            self.click_regions = []
            pitch_rect = self.draw_pitch_area()
            _pitch_rect, image_rect = self._get_pitch_rects()
            self.last_pitch_image_rect = image_rect.copy()
            self._draw_pitch_overlay_info(image_rect)
            self._draw_ball_marker(image_rect)
            self._draw_mid_strategy_panel(pitch_rect, image_rect)
            self._draw_pitch_bottom_controls(pitch_rect, image_rect)
            if not self.report_popup_visible:
                self._handle_strategy_gesture_selection()
                self._handle_pitch_control_gesture()

            for player in self.players:
                player.draw(self.screen, self.font_small)

            if self.hand_detected:
                color = (100, 255, 100) if self.is_grabbing else (255, 255, 100)
                pygame.draw.circle(self.screen, color, (self.hand_x, self.hand_y), 8, 2)
                pygame.draw.circle(self.screen, color, (self.hand_x, self.hand_y), 3)

            self._draw_right_panel(pitch_rect)
            self.draw_status_info()
            self._draw_report_popup()

            pygame.display.flip()
            self.clock.tick(gfi.FPS)

        self._stop_camera_thread()
        pygame.quit()


def main() -> None:
    app = FusionPygameApp()
    app.run()


if __name__ == "__main__":
    main()
