import math
import os
import time
import urllib.request
import ssl
import re
import sys
import random
import itertools
import faulthandler

import cv2
import numpy as np
import pygame
from .paths import (
    FORMATIONS_MD_PATH,
    GROUND_IMAGE_PATHS,
    MODEL_PATH,
    PLAYER_AVATAR_DIR,
)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except Exception as exc:
    raise RuntimeError(
        "MediaPipe 未正确安装或版本不兼容。"
        "请先执行: pip uninstall mediapipe -y && pip install mediapipe"
    ) from exc

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

if os.environ.get("VT_FAULTHANDLER"):
    faulthandler.enable(all_threads=True)

# 窗口和颜色常量
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700
FPS = 60

BACKGROUND_GRAY = (40, 40, 40)
SIDEBAR_BG = (28, 28, 28)
SIDEBAR_PANEL = (34, 34, 34)
SIDEBAR_HIGHLIGHT = (62, 62, 62)
TEXT_PRIMARY = (235, 235, 235)
TEXT_MUTED = (190, 190, 190)
PLAYER_COLOR = (142, 202, 230)
GOALKEEPER_COLOR = (255, 210, 90)
SELECTED_PLAYER_COLOR = (255, 100, 100)
DRAGGING_PLAYER_COLOR = (100, 255, 100)


PIL_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
    "C:/Windows/Fonts/simsun.ttc",  # 宋体
    "C:/Windows/Fonts/simhei.ttf",  # 黑体
    "C:/Windows/Fonts/msyhbd.ttc",  # 微软雅黑粗体
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/Library/Fonts/Microsoft YaHei.ttf",
]

TEXT_COLOR = (255, 255, 255)
SELECT_THRESHOLD = 30  # 像素距离阈值
PLAYER_RADIUS = 17
SELECTED_RADIUS = 17
DRAGGING_RADIUS = 22
AVATAR_DIAMETER_FACTOR = 2.4
TEAM_A_AVATAR_FILES = [f"player{i}.png" for i in range(1, 12)]
TEAM_B_AVATAR_FILES = [f"player{i}.png" for i in range(12, 23)]
PENALTY_AREA_DEPTH_RATIO = 16.5 / 105.0

PYGAME_FONT_PATHS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/Library/Fonts/Microsoft YaHei.ttf",
    "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
    "C:/Windows/Fonts/simsun.ttc",  # 宋体
    "C:/Windows/Fonts/simhei.ttf",  # 黑体
    "C:/Windows/Fonts/msyhbd.ttc",  # 微软雅黑粗体
]

_pygame_font_warned = False

def load_ground_image():
    """加载球场背景图片"""
    for path in GROUND_IMAGE_PATHS:
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return img
    raise FileNotFoundError("ground.jpg or ground.png not found")

def compute_image_rect(area_w, area_h, img_w, img_h, margin):
    """计算图像在区域内的适配矩形"""
    max_w = area_w - margin * 2
    max_h = area_h - margin * 2
    ratio = img_w / img_h
    width = max_w
    height = int(width / ratio)
    if height > max_h:
        height = max_h
        width = int(height * ratio)
    x0 = margin
    y0 = (area_h - height) // 2
    return x0, y0, width, height

_pil_font_cache = {}
_pil_font_path = None

def _get_pil_font(size):
    """获取PIL字体"""
    global _pil_font_path
    if not PIL_AVAILABLE:
        return None
    if _pil_font_path is None:
        for path in PIL_FONT_CANDIDATES:
            if os.path.exists(path):
                _pil_font_path = path
                break
    if _pil_font_path is None:
        return None
    if size not in _pil_font_cache:
        _pil_font_cache[size] = ImageFont.truetype(_pil_font_path, size=size)
    return _pil_font_cache[size]

def draw_text(img, text, org, font_scale=0.6, color=TEXT_COLOR, thickness=1):
    """在图像上绘制文本"""
    if PIL_AVAILABLE:
        font_size = max(12, int(24 * font_scale))
        font = _get_pil_font(font_size)
        if font:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text(org, text, font=font, fill=color[::-1])
            img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

def get_font(size, bold=False):
    """获取pygame字体"""
    candidates = [
        "Microsoft YaHei",
        "SimHei", 
        "PingFang SC",
        "STHeiti",
        "Noto Sans CJK SC",
    ]
    path = None
    for font_path in PYGAME_FONT_PATHS:
        if os.path.exists(font_path):
            path = font_path
            break
    if not path:
        for name in candidates:
            path = pygame.font.match_font(name)
            if path:
                break
    if not path:
        global _pygame_font_warned
        if not _pygame_font_warned:
            print("警告: 未找到可用中文字体，可能导致中文无法显示。")
            _pygame_font_warned = True
    font = pygame.font.Font(path, size) if path else pygame.font.SysFont(None, size)
    font.set_bold(bold)
    return font

def parse_formations_md(path):
    """解析阵型Markdown文件"""
    formations = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return formations

    current = None
    buffer = []
    for line in lines:
        if line.startswith("## "):
            if current and buffer:
                formations[current] = buffer[:]
            current = line.replace("## ", "").strip()
            buffer = []
            continue
        if current:
            stripped = line.lstrip()
            if stripped.startswith("- "):
                indent = len(line) - len(stripped)
                text = stripped[2:].strip()
                level = "sub" if indent >= 2 else "main"
                buffer.append((level, text))
                continue
            if line.strip() == "" and buffer:
                buffer.append(("blank", ""))
    if current and buffer:
        formations[current] = buffer[:]
    return formations

def parse_formation_numbers(name):
    """从阵型名称提取数字"""
    parts = re.findall(r"\d+", name)
    return [int(p) for p in parts]

def roles_for_rows(rows):
    """根据行数确定角色类型"""
    if len(rows) == 3:
        return ["DEF", "MID", "FWD"]
    if len(rows) == 4:
        if rows == [4, 2, 3, 1]:
            return ["DEF", "DM", "AM", "FWD"]
        return ["DEF", "MID", "MID", "FWD"]
    if len(rows) == 5:
        return ["DEF", "DM", "MID", "AM", "FWD"]
    return ["DEF"] + ["MID"] * (len(rows) - 2) + ["FWD"]

def labels_for_role(role, count):
    """为角色生成中文标签"""
    mappings = {
        "DEF": {
            3: ["左中卫", "中卫", "右中卫"],
            4: ["左后卫", "左中卫", "右中卫", "右后卫"],
            5: ["左翼卫", "左中卫", "中卫", "右中卫", "右翼卫"],
        },
        "DM": {1: ["后腰"], 2: ["左后腰", "右后腰"]},
        "MID": {
            1: ["中场"],
            2: ["左中场", "右中场"],
            3: ["左中场", "中场", "右中场"],
            4: ["左前卫", "左中场", "右中场", "右前卫"],
        },
        "AM": {1: ["前腰"], 2: ["左前腰", "右前腰"], 3: ["左前腰", "前腰", "右前腰"]},
        "FWD": {
            1: ["中锋"],
            2: ["左边锋", "右边锋"],
            3: ["左边锋", "中锋", "右边锋"],
        },
    }
    
    mapping = mappings.get(role, {})
    return mapping.get(count, [f"{role}{i+1}" for i in range(count)])


def lane_y_ratios(count):
    """按人数返回更贴近比赛站位的纵向分布（0~1）"""
    presets = {
        1: [0.50],
        2: [0.32, 0.68],
        3: [0.23, 0.50, 0.77],
        4: [0.16, 0.38, 0.62, 0.84],
        5: [0.12, 0.31, 0.50, 0.69, 0.88],
    }
    if count in presets:
        return presets[count]
    return [(i + 1) / (count + 1) for i in range(count)]

class Player:
    """球员类"""
    def __init__(self, x, y, label, role, avatar_surface=None, team="A"):
        self.x = x
        self.y = y
        self.label = label
        self.role = role
        self.avatar_surface = avatar_surface
        self.team = team
        self.selected = False
        self.dragging = False
        self.hovered = False
        self.original_x = x
        self.original_y = y
        self.out_of_bounds = False  # 出界标记
        self._avatar_cache = {}
        
    def set_position(self, x, y, pitch_bounds):
        """设置球员位置，实现预警式边界限制"""
        min_x, min_y, max_x, max_y = pitch_bounds
        
        # 设置预警距离（提前10像素开始限制）
        warning_distance = 10
        warn_min_x = min_x + warning_distance
        warn_min_y = min_y + warning_distance
        warn_max_x = max_x - warning_distance
        warn_max_y = max_y - warning_distance
        
        # 检查是否进入预警区域
        in_warning_zone = not (warn_min_x <= x <= warn_max_x and warn_min_y <= y <= warn_max_y)
        
        # 检查是否真正出界
        self.out_of_bounds = not (min_x <= x <= max_x and min_y <= y <= max_y)
        
        # 预警式限制：在预警区域内就开始限制向外出移动
        if in_warning_zone and (x < min_x or x > max_x or y < min_y or y > max_y):
            # 在预警区试图出界，保持当前位置
            return False
        
        # 完全出界也阻止
        if self.out_of_bounds:
            return False
        
        # 正常更新位置
        self.x = x
        self.y = y
        return True
        
    def distance_to(self, x, y):
        """计算到指定点的距离"""
        return math.sqrt((self.x - x)**2 + (self.y - y)**2)
    
    def is_hovered(self, x, y):
        """检查是否被悬停"""
        return self.distance_to(x, y) <= SELECT_THRESHOLD
    
    def draw(self, surface, font):
        """绘制球员"""
        radius = PLAYER_RADIUS
        border_color = (30, 30, 30)
        border_width = 2
        if self.dragging:
            radius = DRAGGING_RADIUS
            border_color = DRAGGING_PLAYER_COLOR
            border_width = 3
        elif self.hovered:
            border_color = SELECTED_PLAYER_COLOR
            border_width = 3

        draw_pos = (int(self.x), int(self.y))

        # 优先绘制头像（A队），缺失时回退圆点
        if self.avatar_surface is not None:
            diameter = max(18, int(radius * AVATAR_DIAMETER_FACTOR))
            if diameter not in self._avatar_cache:
                self._avatar_cache[diameter] = pygame.transform.smoothscale(
                    self.avatar_surface, (diameter, diameter)
                )
            avatar = self._avatar_cache[diameter]
            avatar_rect = avatar.get_rect(center=draw_pos)
            surface.blit(avatar, avatar_rect)
            ring_radius = max(radius, diameter // 2)
            pygame.draw.circle(surface, border_color, draw_pos, ring_radius, border_width)
            pygame.draw.circle(surface, (30, 30, 30), draw_pos, ring_radius + 1, 1)
        else:
            fill_color = PLAYER_COLOR if not self.dragging else DRAGGING_PLAYER_COLOR
            pygame.draw.circle(surface, fill_color, draw_pos, radius)
            pygame.draw.circle(surface, border_color, draw_pos, radius, border_width)
        
        # 绘制标签
        text = font.render(self.label, True, (245, 245, 245))
        label_offset = radius + 14
        text_rect = text.get_rect(center=(int(self.x), int(self.y - label_offset)))
        if text_rect.top < 2:
            text_rect = text.get_rect(center=(int(self.x), int(self.y + label_offset)))
        surface.blit(text, text_rect)

class GestureFormationApp:
    """手势阵型交互应用"""
    def __init__(self):
        # 初始化pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("手势-阵型交互系统")
        self.clock = pygame.time.Clock()
        
        # 初始化字体
        self.font_title = get_font(22, bold=True)
        self.font_body = get_font(18)
        self.font_small = get_font(16)
        
        # 加载数据
        self.formations = parse_formations_md(FORMATIONS_MD_PATH)
        self.formation_names = list(self.formations.keys()) if self.formations else ["4-4-2", "4-3-3", "4-2-3-1"]
        self.default_formation = self.pick_default_formation()
        self.selected_formation_a = self.default_formation
        self.selected_formation_b = self.default_formation
        self.team_a_avatars = self.load_team_avatars(TEAM_A_AVATAR_FILES)
        self.team_b_avatars = self.load_team_avatars(TEAM_B_AVATAR_FILES)
        
        # 球场边界（初始默认值，基于图像实际边界）
        player_radius = DRAGGING_RADIUS + 2
        # 默认使用标准比例计算（假设图像为16:9）
        default_img_w, default_img_h = 800, 450
        area_w = WINDOW_WIDTH - 320 - 20
        area_h = WINDOW_HEIGHT - 20
        margin = 10
        img_x, img_y, img_width, img_height = compute_image_rect(area_w, area_h, default_img_w, default_img_h, margin)
        self.pitch_bounds = (
            10 + img_x + player_radius,
            10 + img_y + player_radius,
            10 + img_x + img_width - player_radius,
            10 + img_y + img_height - player_radius
        )
        self.out_of_bounds_message = ""  # 出界提示信息
        self.out_of_bounds_timer = 0   # 出界提示计时器
        
        # 加载背景图像
        try:
            self.ground_img_cv = load_ground_image()
            self.ground_size = self.ground_img_cv.shape[1], self.ground_img_cv.shape[0]
            # 转换为pygame Surface
            self.ground_img = pygame.surfarray.make_surface(
                cv2.cvtColor(self.ground_img_cv, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
            )
        except FileNotFoundError:
            self.ground_img_cv = None
            self.ground_img = None
            self.ground_size = (800, 500)
        
        # 初始化球员
        self.players = []
        self.create_players()
        
        # 移除侧边栏相关代码
        self.show_sidebar = False
        
        # 手势相关
        self.hand_detector = self.initialize_hand_detector()
        
        # 摄像头初始化（添加详细错误处理）
        print("初始化摄像头...")
        self.cap = None
        camera_initialized = False
        
        # 尝试多个摄像头索引
        camera_indices = [0, 1, 2, -1]  # -1表示默认摄像头
        for idx in camera_indices:
            try:
                if idx == -1:
                    print("尝试默认摄像头...")
                    cap_test = cv2.VideoCapture()
                else:
                    print(f"尝试摄像头索引 {idx}...")
                    cap_test = cv2.VideoCapture(idx)
                
                if cap_test.isOpened():
                    ret, frame = cap_test.read()
                    if ret and frame is not None:
                        print(f"✓ 摄像头 {idx if idx != -1 else '默认'} 初始化成功")
                        self.cap = cap_test
                        camera_initialized = True
                        break
                    else:
                        print(f"✗ 摄像头 {idx if idx != -1 else '默认'} 无法读取帧")
                        cap_test.release()
                else:
                    print(f"✗ 摄像头 {idx if idx != -1 else '默认'} 无法打开")
                    cap_test.release()
            except Exception as e:
                print(f"✗ 摄像头 {idx if idx != -1 else '默认'} 初始化异常: {e}")
                if 'cap_test' in locals():
                    cap_test.release()
        
        if not camera_initialized:
            print("✗ 所有摄像头初始化失败")
            print("程序将在无摄像头模式下运行（仅阵型调整功能可用）")
            self.cap = None
        
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.1  # 100ms冷却时间（提高响应速度）
        self.lost_tracking_count = 0  # 丢失跟踪计数
        self.max_lost_frames = 3  # 允许的最大丢失帧数
        
        # 交互状态
        self.selected_player = None
        self.dragging_player = None
        self.hand_x = 0
        self.hand_y = 0
        self.hand_detected = False
        self.hand_keypoints = None
        
        # 阵型确认相关
        self.confirmation_message = ""
        self.confirmation_timer = 0
        self.last_ok_gesture_time = 0
        self.ok_gesture_cooldown = 1.0  # 1秒冷却时间，避免重复触发
        self.ok_hold_duration = 1.0  # OK手势需要保持的时间
        self.ok_hold_start_time = None
        self.ok_hold_elapsed = 0.0
        self.ok_confirmed_wait_release = False
        
        # 阵型调整显示信息
        self.last_adjustment_from = None
        self.last_adjustment_to = None
        
        # 平滑移动相关
        self.smooth_movement_active = False
        self.movement_targets = {}  # 存储每个球员的目标位置
        self.movement_speed = 3.0   # 基础移动速度（像素/帧）
        self.movement_duration_seconds = 1.5  # 阵型调整总时长
        self.movement_completed_callback = None
        self.animation_progress = {}  # 存储每个球员的动画进度(0.0-1.0)
        
        # B队自动换阵：每10秒随机切换一次（与A队独立）
        self.team_b_auto_switch_interval = 10.0
        self.last_team_b_auto_switch_time = time.time()
        self.pending_team_b_target_formation = None
        # A队手动调整后待确认锁：未OK前冻结B队自动换阵
        self.a_position_dirty_pending_ok = False
        
    def initialize_hand_detector(self):
        """初始化手势检测器"""
        print("开始初始化手势检测器...")
        
        if not os.path.exists(MODEL_PATH):
            print(f"模型文件 {MODEL_PATH} 不存在，尝试下载...")
            try:
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                with urllib.request.urlopen(url, context=context) as response:
                    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                    with open(MODEL_PATH, 'wb') as f:
                        f.write(response.read())
                print("模型下载完成")
            except Exception as e:
                print(f"模型下载失败: {e}")
                print("将继续尝试使用现有配置运行")
                
        try:
            print(f"加载模型文件: {MODEL_PATH}")
            if not os.path.exists(MODEL_PATH):
                print(f"错误: 模型文件 {MODEL_PATH} 不存在")
                return None
                
            base_options = mp_python.BaseOptions(
                model_asset_path=MODEL_PATH,
                delegate=mp_python.BaseOptions.Delegate.CPU
            )
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.3,  # 降低阈值提高检测率
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3
            )
            print("创建手势检测器...")
            detector = mp_vision.HandLandmarker.create_from_options(options)
            print("✓ 手势检测器初始化成功")
            return detector
        except Exception as e:
            print(f"✗ 手势检测器初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def pick_default_formation(self):
        """优先选择4-4-2作为默认阵型"""
        preferred = ["4-4-2", "4-4-2：经典平衡，双前锋。"]
        for name in preferred:
            if name in self.formation_names:
                return name
        for name in self.formation_names:
            nums = parse_formation_numbers(name)
            if nums == [4, 4, 2]:
                return name
        return self.formation_names[0] if self.formation_names else "4-4-2"

    def load_team_avatars(self, avatar_files):
        """加载指定队伍头像"""
        avatars = []
        for filename in avatar_files:
            path = os.path.join(PLAYER_AVATAR_DIR, filename)
            if not os.path.exists(path):
                avatars.append(None)
                continue
            try:
                img = pygame.image.load(path).convert_alpha()
                avatars.append(img)
            except Exception as e:
                print(f"加载头像失败: {path} ({e})")
                avatars.append(None)
        return avatars

    def _compute_pitch_bounds(self):
        """计算球场边界（基于实际图像边界）"""
        _, image_rect = self._get_pitch_rects()
        player_radius = DRAGGING_RADIUS + 2
        self.pitch_bounds = (
            image_rect.left + player_radius,
            image_rect.top + player_radius,
            image_rect.right - player_radius,
            image_rect.bottom - player_radius,
        )

    def _get_pitch_rects(self):
        """返回球场容器矩形与等比背景矩形"""
        sw, sh = self.screen.get_size()
        sidebar_w = max(320, int(sw * 0.28))
        pitch_rect = pygame.Rect(10, 10, sw - sidebar_w - 20, sh - 20)

        img_w, img_h = self.ground_size
        margin = 10
        fit_x, fit_y, fit_w, fit_h = compute_image_rect(
            pitch_rect.width, pitch_rect.height, img_w, img_h, margin
        )
        image_rect = pygame.Rect(
            pitch_rect.x + fit_x,
            pitch_rect.y + fit_y,
            fit_w,
            fit_h,
        )
        return pitch_rect, image_rect

    def _compute_interleaved_line_positions(self, rows_a, rows_b):
        """计算A/B两队交错且全局等间距的列X位置（考虑列数不一致）"""
        min_x, _min_y, max_x, _max_y = self.pitch_bounds
        pitch_w = max_x - min_x
        margin_x = pitch_w * 0.06
        count_a = len(rows_a)
        count_b = len(rows_b)
        total = count_a + count_b
        if total <= 0:
            return [], []

        # 使用“左禁区线到右禁区线”之间的区域作为阵线横向范围
        penalty_depth = pitch_w * PENALTY_AREA_DEPTH_RATIO
        inner_pad = max(8.0, pitch_w * 0.01)
        span_left = min_x + penalty_depth + inner_pad
        span_right = max_x - penalty_depth - inner_pad
        if span_right <= span_left:
            # 极端窗口下兜底
            span_left = min_x + max(margin_x * 1.2, pitch_w * 0.18)
            span_right = max_x - max(margin_x * 1.2, pitch_w * 0.18)

        if total == 1:
            slots = [(span_left + span_right) / 2.0]
        else:
            gap = (span_right - span_left) / (total - 1)
            slots = [span_left + i * gap for i in range(total)]

        # 左到右分配槽位归属：
        # 先尽量AB交错；多余列放回本方侧（A放左端，B放右端）
        pair_count = min(count_a, count_b)
        core = []
        for _ in range(pair_count):
            core.append("A")
            core.append("B")
        extra_a = ["A"] * (count_a - pair_count)
        extra_b = ["B"] * (count_b - pair_count)
        owners = extra_a + core + extra_b

        # owners长度应与slots一致
        owners = owners[:len(slots)]
        a_slot_idx = [i for i, o in enumerate(owners) if o == "A"][:count_a]
        b_slot_idx = [i for i, o in enumerate(owners) if o == "B"][:count_b]

        # A: 防线->锋线 = 左->右
        a_lines = [slots[i] for i in sorted(a_slot_idx)]
        # B: 防线->锋线 = 右->左
        b_lines = [slots[i] for i in sorted(b_slot_idx, reverse=True)]
        return a_lines, b_lines

    def _build_team_players(self, team_name, formation_name, avatars, attack_to_right=True, line_positions=None):
        """为单支球队生成阵型点位"""
        rows = parse_formation_numbers(formation_name)
        if not rows:
            rows = [4, 4, 2]
        roles = roles_for_rows(rows)
        players = []
        player_idx = 0

        min_x, min_y, max_x, max_y = self.pitch_bounds
        pitch_w = max_x - min_x
        pitch_h = max_y - min_y
        margin_x = pitch_w * 0.06
        margin_y = pitch_h * 0.06

        gk_ratio = 0.04
        gk_x = min_x + int(pitch_w * gk_ratio) if attack_to_right else max_x - int(pitch_w * gk_ratio)
        gk_y = (min_y + max_y) // 2
        gk_avatar = avatars[player_idx] if player_idx < len(avatars) else None
        players.append(Player(gk_x, gk_y, "门将", "GK", gk_avatar, team=team_name))
        player_idx += 1

        total_lines = len(rows)
        penalty_depth = pitch_w * PENALTY_AREA_DEPTH_RATIO
        inner_pad = max(8.0, pitch_w * 0.01)
        line_start = min_x + penalty_depth + inner_pad
        line_end = max_x - penalty_depth - inner_pad
        if line_end <= line_start:
            line_start = min_x + margin_x * 1.2
            line_end = min_x + pitch_w - margin_x * 0.8

        for i, count in enumerate(rows):
            if line_positions and i < len(line_positions):
                line_x = line_positions[i]
            else:
                base_x = line_start + (line_end - line_start) * (i + 1) / (total_lines + 1)
                line_x = base_x if attack_to_right else (min_x + max_x - base_x)
            labels = labels_for_role(roles[i] if i < len(roles) else "MID", count)
            y_ratios = lane_y_ratios(count)
            for j in range(count):
                y_ratio = y_ratios[j] if j < len(y_ratios) else (j + 1) / (count + 1)
                y = min_y + margin_y + (pitch_h - 2 * margin_y) * y_ratio
                # 两队头像交错：B队纵向轻微偏移，避免完全重合感
                if team_name == "B":
                    y += 6 if (j % 2 == 0) else -6
                label = labels[j] if j < len(labels) else f"P{j + 1}"
                avatar = avatars[player_idx] if player_idx < len(avatars) else None
                players.append(Player(line_x, y, label, roles[i], avatar, team=team_name))
                player_idx += 1

        return players

    def create_players(self):
        """创建A/B两队球员点位，并交错组织绘制顺序"""
        self.players = []
        self._compute_pitch_bounds()
        rows_a = parse_formation_numbers(self.selected_formation_a) or [4, 4, 2]
        rows_b = parse_formation_numbers(self.selected_formation_b) or [4, 4, 2]
        line_x_a, line_x_b = self._compute_interleaved_line_positions(rows_a, rows_b)

        team_a_players = self._build_team_players(
            team_name="A",
            formation_name=self.selected_formation_a,
            avatars=self.team_a_avatars,
            attack_to_right=True,
            line_positions=line_x_a,
        )
        team_b_players = self._build_team_players(
            team_name="B",
            formation_name=self.selected_formation_b,
            avatars=self.team_b_avatars,
            attack_to_right=False,
            line_positions=line_x_b,
        )

        # 交错顺序：A1, B1, A2, B2...
        max_len = max(len(team_a_players), len(team_b_players))
        for i in range(max_len):
            if i < len(team_a_players):
                self.players.append(team_a_players[i])
            if i < len(team_b_players):
                self.players.append(team_b_players[i])
    
    def draw_hand_skeleton(self, frame, keypoints):
        """在图像上绘制手部骨架"""
        if not keypoints or 'landmarks' not in keypoints:
            return frame
            
        landmarks = keypoints['landmarks']
        h, w = frame.shape[:2]
        
        # 定义手部连接关系
        connections = [
            # 手掌
            (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
            (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
            # 手指间连接
            (5, 9), (9, 13), (13, 17),  # 手背连接
        ]
        
        # 绘制连接线
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                
                start_x = int(start_point.x * w)
                start_y = int(start_point.y * h)
                end_x = int(end_point.x * w)
                end_y = int(end_point.y * h)
                
                # 根据抓取状态改变线条颜色
                if keypoints.get('is_pinching', False):
                    color = (0, 255, 0)  # 绿色表示捏合
                elif keypoints.get('fingers_curled', False):
                    color = (0, 255, 255)  # 黄绿色表示弯曲
                else:
                    color = (255, 255, 255)  # 白色表示普通状态
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)
        
        # 绘制关键点
        for i, landmark in enumerate(landmarks):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            
            # 根据点的重要性使用不同颜色和大小
            if i in [0, 4, 8, 12, 16, 20]:  # 关键点：手腕和指尖
                color = (0, 0, 255)  # 红色
                radius = 5
            else:
                color = (255, 0, 0)  # 蓝色
                radius = 3
            
            cv2.circle(frame, (x, y), radius, color, -1)
            cv2.circle(frame, (x, y), radius, (255, 255, 255), 1)
        
        return frame
    
    def detect_ok_gesture(self, landmarks):
        """检测OK手势（拇指和食指形成圆形）"""
        try:
            # 获取关键点
            thumb_tip = landmarks[4]      # 拇指尖端
            index_tip = landmarks[8]      # 食指尖端
            index_mcp = landmarks[5]      # 食指根部
            middle_mcp = landmarks[9]     # 中指根部
            pinky_mcp = landmarks[17]     # 小指根部
            wrist = landmarks[0]          # 手腕
            
            # 计算拇指和食指指尖之间的距离
            tip_distance = math.sqrt(
                (thumb_tip.x - index_tip.x)**2 +
                (thumb_tip.y - index_tip.y)**2
            )

            # 动态尺度：使用掌宽/掌长估计手的尺度，避免固定阈值在不同距离下失效
            palm_width = math.sqrt(
                (index_mcp.x - pinky_mcp.x)**2 +
                (index_mcp.y - pinky_mcp.y)**2
            )
            palm_length = math.sqrt(
                (wrist.x - middle_mcp.x)**2 +
                (wrist.y - middle_mcp.y)**2
            )
            hand_scale = max(palm_width, palm_length, 1e-6)

            # 计算手掌中心作为参考点（更稳定）
            palm_center_x = (wrist.x + index_mcp.x + middle_mcp.x + pinky_mcp.x) / 4
            palm_center_y = (wrist.y + index_mcp.y + middle_mcp.y + pinky_mcp.y) / 4

            # 检查其他手指是否伸展
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]

            def _dist_to_palm(tip):
                return math.sqrt((tip.x - palm_center_x)**2 + (tip.y - palm_center_y)**2)

            middle_extension = _dist_to_palm(middle_tip)
            ring_extension = _dist_to_palm(ring_tip)
            pinky_extension = _dist_to_palm(pinky_tip)

            # 动态阈值（基于手的尺度）
            tip_threshold = hand_scale * 0.40          # 拇指/食指指尖接近阈值
            extension_threshold = hand_scale * 0.75    # 其他手指伸展阈值

            extended_count = sum(
                ext > extension_threshold
                for ext in (middle_extension, ring_extension, pinky_extension)
            )

            # OK手势判定（更宽松但抑制误触）：
            # 1) 拇指与食指指尖足够接近
            # 2) 其余至少两指伸展（避免与握拳/抓取混淆）
            is_ok = (tip_distance < tip_threshold) and (extended_count >= 2)

            if os.environ.get("VT_DEBUG_OK"):
                print(
                    "OK调试: tip={:.3f} thr={:.3f} scale={:.3f} "
                    "mid={:.3f} ring={:.3f} pinky={:.3f} ext_thr={:.3f} ext_cnt={}".format(
                        tip_distance,
                        tip_threshold,
                        hand_scale,
                        middle_extension,
                        ring_extension,
                        pinky_extension,
                        extension_threshold,
                        extended_count,
                    )
                )

            return is_ok
        except Exception as e:
            print(f"OK手势检测错误: {e}")
            return False
    
    def detect_gesture(self, frame):
        """检测手势"""
        if not self.hand_detector:
            return None, None, None
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.hand_detector.detect(mp_image)
            
            if detection_result.hand_landmarks:
                landmarks = detection_result.hand_landmarks[0]
                
                # 获取关键点坐标
                thumb_tip = landmarks[4]      # 拇指尖端
                index_tip = landmarks[8]      # 食指尖端
                middle_tip = landmarks[12]    # 中指尖端
                ring_tip = landmarks[16]      # 无名指尖端
                pinky_tip = landmarks[20]     # 小指尖端
                
                wrist = landmarks[0]          # 手腕
                thumb_mcp = landmarks[2]      # 拇指根部
                index_mcp = landmarks[5]      # 食指根部
                
                # 计算多种抓取指标
                # 1. 拇指-食指捏合距离
                pinch_distance = math.sqrt(
                    (thumb_tip.x - index_tip.x)**2 + 
                    (thumb_tip.y - index_tip.y)**2
                )
                
                # 2. 手指弯曲程度（通过指尖到手掌根部的距离）
                palm_center_x = (index_mcp.x + thumb_mcp.x) / 2
                palm_center_y = (index_mcp.y + thumb_mcp.y) / 2
                
                # 计算各手指的伸展程度（相对于手掌中心）
                index_extension = math.sqrt(
                    (index_tip.x - palm_center_x)**2 + 
                    (index_tip.y - palm_center_y)**2
                )
                middle_extension = math.sqrt(
                    (middle_tip.x - palm_center_x)**2 + 
                    (middle_tip.y - palm_center_y)**2
                )
                ring_extension = math.sqrt(
                    (ring_tip.x - palm_center_x)**2 + 
                    (ring_tip.y - palm_center_y)**2
                )
                pinky_extension = math.sqrt(
                    (pinky_tip.x - palm_center_x)**2 + 
                    (pinky_tip.y - palm_center_y)**2
                )
                
                # 综合判定抓取状态
                # 半握：拇指接近食指，大部分手指有一定弯曲
                # 全握：拇指接触食指，手指明显弯曲
                pinch_threshold = 0.08  # 捏合阈值
                extension_threshold = 0.15  # 伸展阈值（越小表示越弯曲）
                
                is_pinching = pinch_distance < pinch_threshold
                fingers_curled = (
                    index_extension < extension_threshold and
                    middle_extension < extension_threshold and
                    (ring_extension < extension_threshold * 1.2 or 
                     pinky_extension < extension_threshold * 1.2)
                )
                
                # 改进的抓取判定：大部分手指弯曲即可
                # 要求至少3个手指明显弯曲（更符合实际握拳习惯）
                fingers_curled_count = 0
                if index_extension < extension_threshold:
                    fingers_curled_count += 1
                if middle_extension < extension_threshold:
                    fingers_curled_count += 1
                if ring_extension < extension_threshold:
                    fingers_curled_count += 1
                if pinky_extension < extension_threshold:
                    fingers_curled_count += 1
                
                # 判定为抓取：至少3个手指弯曲
                is_grabbing = fingers_curled_count >= 3
                
                # 检测OK手势（拇指和食指形成圆形）
                is_ok_gesture = self.detect_ok_gesture(landmarks)
                
                # 使用手掌关节中心点作为光标定位（更稳定）
                # 采用手腕、食指根部、小指根部的平均位置
                pinky_mcp = landmarks[17]     # 小指根部
                
                hand_center_x = (wrist.x + index_mcp.x + pinky_mcp.x) / 3.0
                hand_center_y = (wrist.y + index_mcp.y + pinky_mcp.y) / 3.0
                
                # 返回关键点用于绘制骨架
                key_points = {
                    'landmarks': landmarks,
                    'pinch_distance': pinch_distance,
                    'is_pinching': is_pinching,
                    'fingers_curled': fingers_curled,
                    'fingers_curled_count': fingers_curled_count,
                    'is_ok_gesture': is_ok_gesture,
                    'index_extension': index_extension,
                    'middle_extension': middle_extension,
                    'ring_extension': ring_extension,
                    'pinky_extension': pinky_extension
                }
                
                return is_grabbing, (hand_center_x, hand_center_y), key_points
            
            return None, None, None
        except Exception as e:
            print(f"手势检测错误: {e}")
            return None, None, None
    
    def map_hand_to_screen(self, hand_coords):
        """将手部坐标映射到屏幕坐标"""
        if not hand_coords:
            return None
            
        hand_x_norm, hand_y_norm = hand_coords
        
        # 获取屏幕尺寸
        sw, sh = self.screen.get_size()
        sidebar_w = max(320, int(sw * 0.28))
        playable_w = sw - sidebar_w - 20
        playable_h = sh - 20
        
        # 映射手部坐标到可玩区域
        screen_x = 20 + playable_w * (1 - hand_x_norm)  # 镜像X轴
        screen_y = 10 + playable_h * hand_y_norm
        
        return int(screen_x), int(screen_y)
    
    def handle_ok_gesture_confirmation(self):
        """处理OK手势确认阵型"""
        if (self.hand_keypoints and
            self.hand_keypoints.get('is_ok_gesture', False) and
            self.hand_detected):
            
            current_time = time.time()
            
            # 已触发但未松开，等待释放
            if self.ok_confirmed_wait_release:
                return
            
            # 记录OK手势开始时间
            if self.ok_hold_start_time is None:
                self.ok_hold_start_time = current_time
                self.ok_hold_elapsed = 0.0
                return
            
            self.ok_hold_elapsed = current_time - self.ok_hold_start_time
            
            # 持续时间达到阈值并通过冷却时间，才触发
            if (self.ok_hold_elapsed >= self.ok_hold_duration and
                current_time - self.last_ok_gesture_time > self.ok_gesture_cooldown):
                
                # 识别当前阵型
                recognized_formation = self.recognize_formation()
                print(f"识别到的当前阵型: {recognized_formation}")
                
                # 查找对应的标准阵型
                target_formation = self.find_closest_standard_formation(recognized_formation)
                
                if target_formation:
                    # 记录转换前阵型
                    prev_formation = self.selected_formation_a
                    
                    # 如果前后阵型一致且无待确认手动改动，不触发调整流程
                    if prev_formation == target_formation and not self.a_position_dirty_pending_ok:
                        self.confirmation_message = f"无需调整（已是 {target_formation}）"
                        self.confirmation_timer = 180  # 显示3秒
                        self.last_ok_gesture_time = current_time
                        print(f"阵型未变化，跳过调整: {target_formation}")
                        
                        # 触发后等待松手
                        self.ok_confirmed_wait_release = True
                        self.ok_hold_start_time = None
                        self.ok_hold_elapsed = 0.0
                        return
                    
                    # 显示确认消息与调整过程（OK后同步A/B，但不强制同阵型）
                    self.last_adjustment_from = prev_formation
                    self.last_adjustment_to = target_formation
                    b_target = self.pending_team_b_target_formation or self.selected_formation_b
                    self.confirmation_message = f"确认同步: A→{target_formation} | B→{b_target}"
                    self.confirmation_timer = 240  # 显示4秒
                    self.last_ok_gesture_time = current_time
                    
                    # 执行阵型调整：同步A队与B队（B优先使用待执行目标）
                    success = self.adjust_formation_to_standard(
                        target_formation,
                        team_name="both",
                        target_formation_b=b_target,
                    )
                    if success:
                        print(f"A/B阵型已同步调整: A={target_formation}, B={b_target}")
                        self.a_position_dirty_pending_ok = False
                        self.pending_team_b_target_formation = None
                        self.last_team_b_auto_switch_time = current_time
                    else:
                        print(f"A/B阵型同步调整失败: {target_formation}")
                else:
                    # 无法识别或找不到对应标准阵型
                    self.confirmation_message = "没有匹配阵型"
                    self.confirmation_timer = 120  # 显示2秒
                    self.last_ok_gesture_time = current_time
                    print(f"无法找到 {recognized_formation} 对应的标准阵型")
                
                # 触发后等待松手
                self.ok_confirmed_wait_release = True
                self.ok_hold_start_time = None
                self.ok_hold_elapsed = 0.0
        else:
            # OK手势取消，重置计时
            self.ok_hold_start_time = None
            self.ok_hold_elapsed = 0.0
            self.ok_confirmed_wait_release = False
    
    def update_smooth_movement(self):
        """更新缓动动画移动"""
        if not self.smooth_movement_active or not self.movement_targets:
            return
            
        movement_completed = True
        total_frames = max(int(FPS * self.movement_duration_seconds), 1)
        animation_speed = 1.0 / total_frames  # 动画速度(每帧进度增量)
        
        for i, target_info in self.movement_targets.items():
            if i >= len(self.players):
                continue
                
            player = self.players[i]
            target_x = target_info['target_x']
            target_y = target_info['target_y']
            start_x = target_info['start_x']
            start_y = target_info['start_y']
            
            # 更新动画进度
            if i in self.animation_progress:
                self.animation_progress[i] += animation_speed
                progress = min(1.0, self.animation_progress[i])
            else:
                progress = 1.0
                self.animation_progress[i] = 1.0
            
            # 应用缓动函数（ease-out cubic）
            eased_progress = 1 - pow(1 - progress, 3)  # 立方缓出，开始快结束慢
            
            # 计算新位置
            new_x = start_x + (target_x - start_x) * eased_progress
            new_y = start_y + (target_y - start_y) * eased_progress
            
            # 更新球员位置
            player.set_position(new_x, new_y, self.pitch_bounds)
            
            # 保持状态
            if hasattr(self, 'dragging_player') and self.dragging_player == player:
                player.dragging = True
            elif hasattr(self, 'selected_player') and self.selected_player == player:
                player.selected = True
            
            # 检查是否完成动画
            if progress < 1.0:
                movement_completed = False
        
        # 如果所有动画都完成
        if movement_completed:
            self.smooth_movement_active = False
            self.movement_targets.clear()
            self.animation_progress.clear()
            print("动画移动完成")
            
            # 执行回调函数（如果有的话）
            if self.movement_completed_callback:
                self.movement_completed_callback()
                self.movement_completed_callback = None
    
    def handle_gesture_interaction(self):
        """处理手势交互"""
        # 处理确认消息计时器
        if self.confirmation_timer > 0:
            self.confirmation_timer -= 1
        
        if not self.hand_detected:
            for player in self.players:
                player.hovered = False
            # 没有检测到手时增加丢失计数
            self.lost_tracking_count += 1
            
            # 如果丢失帧数超过阈值，才释放状态
            if self.lost_tracking_count > self.max_lost_frames:
                if self.dragging_player:
                    self.dragging_player.dragging = False
                    self.dragging_player = None
                if self.selected_player:
                    self.selected_player.selected = False
                    self.selected_player = None
                self.lost_tracking_count = 0  # 重置计数
            return
        else:
            # 检测到手时重置丢失计数
            self.lost_tracking_count = 0
        
        # 检查是否有球员被选中（仅A队可交互）
        hovered_player = None
        for player in self.players:
            if player.team != "A":
                continue
            if player.is_hovered(self.hand_x, self.hand_y):
                hovered_player = player
                break
        for player in self.players:
            player.hovered = (player == hovered_player and player.team == "A")

        current_time = time.time()
        
        # 检查冷却时间（视觉高亮已更新）
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return
        
        if self.dragging_player:
            # 防御性处理：右侧B队不允许被拖拽
            if self.dragging_player.team != "A":
                self.dragging_player.dragging = False
                self.dragging_player = None
                if self.selected_player and self.selected_player.team != "A":
                    self.selected_player.selected = False
                    self.selected_player = None
                return
            # 正在拖拽球员
            if not self.is_grabbing:  # 松开手势
                self.dragging_player.dragging = False
                self.dragging_player = None
                self.last_gesture_time = current_time
            else:  # 继续拖拽
                # 实现预警式边界限制
                old_x, old_y = self.dragging_player.x, self.dragging_player.y
                success = self.dragging_player.set_position(self.hand_x, self.hand_y, self.pitch_bounds)
                if success and self.dragging_player.team == "A":
                    moved = math.hypot(self.dragging_player.x - old_x, self.dragging_player.y - old_y)
                    if moved > 0.1:
                        self.a_position_dirty_pending_ok = True
                
                # 检查是否进入预警区域或出界
                min_x, min_y, max_x, max_y = self.pitch_bounds
                warning_distance = 10
                warn_min_x = min_x + warning_distance
                warn_min_y = min_y + warning_distance
                warn_max_x = max_x - warning_distance
                warn_max_y = max_y - warning_distance
                
                in_warning_zone = not (warn_min_x <= self.dragging_player.x <= warn_max_x and 
                                     warn_min_y <= self.dragging_player.y <= warn_max_y)
                is_out_of_bounds = self.dragging_player.out_of_bounds
                
                # 如果进入预警区域或出界，显示警告
                if not success or in_warning_zone or is_out_of_bounds:
                    self.show_out_of_bounds_warning()
        else:
            # 没有拖拽中
            if self.is_grabbing and hovered_player:  # 开始抓取
                # 取消之前选中的球员
                if self.selected_player:
                    self.selected_player.selected = False
                
                # 设置新的拖拽状态
                self.selected_player = hovered_player
                self.dragging_player = hovered_player
                self.dragging_player.selected = True
                self.dragging_player.dragging = True
                self.last_gesture_time = current_time
            elif hovered_player and not self.is_grabbing:
                # 悬停效果（仅显示选中状态，不拖拽）
                if self.selected_player != hovered_player:
                    if self.selected_player:
                        self.selected_player.selected = False
                    self.selected_player = hovered_player
                    self.selected_player.selected = True
    
    def draw_pitch_area(self):
        """绘制球场区域"""
        pitch_rect, image_rect = self._get_pitch_rects()
        
        # 绘制背景
        if self.ground_img is not None:
            # 先绘制容器底色，再将背景图等比缩放后居中显示
            pygame.draw.rect(self.screen, (18, 18, 18), pitch_rect)
            scaled_img = pygame.transform.smoothscale(
                self.ground_img, (image_rect.width, image_rect.height)
            )
            self.screen.blit(scaled_img, image_rect)
        else:
            pygame.draw.rect(self.screen, (34, 139, 34), pitch_rect)
        
        # 绘制边界（外容器+实际球场图像区域）
        pygame.draw.rect(self.screen, (255, 255, 255), pitch_rect, 2)
        pygame.draw.rect(self.screen, (210, 210, 210), image_rect, 1)
        
        return pitch_rect

    def draw_camera_info_line(self, text, x, y, color=(220, 220, 220)):
        """在摄像头窗口下方绘制带背景的单行信息"""
        line_surface = self.font_small.render(text, True, color)
        line_rect = line_surface.get_rect(topleft=(x, y))
        bg_rect = line_rect.inflate(8, 6)
        pygame.draw.rect(self.screen, (20, 20, 20), bg_rect, border_radius=4)
        self.screen.blit(line_surface, (x + 4, y + 3))
        return bg_rect.bottom + 4

    def get_formation_intro_lines(self, formation_name):
        """获取阵型介绍文字（完整条目）"""
        if not formation_name or formation_name not in self.formations:
            return []
        items = self.formations.get(formation_name, [])
        lines = []
        for level, text in items:
            if level == "blank":
                continue
            if text:
                lines.append(text)
        return lines
    
    # 移除draw_sidebar方法，不再需要阵型选择界面
    
    def embed_camera_frame(self, frame):
        """将摄像头画面嵌入到pygame窗口"""
        if self.hand_keypoints:
            frame = self.draw_hand_skeleton(frame, self.hand_keypoints)
        
        # 调整画面大小
        camera_width = 240
        camera_height = 180
        display_frame = cv2.resize(frame, (camera_width, camera_height))
        
        # 转换为pygame surface
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        
        # 绘制到屏幕右上角
        sw, sh = self.screen.get_size()
        camera_x = sw - camera_width - 20
        camera_y = 20
        
        # 绘制边框
        pygame.draw.rect(self.screen, (70, 70, 70), 
                        (camera_x - 2, camera_y - 2, camera_width + 4, camera_height + 4), 2)
        
        # 绘制摄像头画面
        self.screen.blit(frame_surface, (camera_x, camera_y))
        
        # 绘制标题
        title_surface = self.font_small.render("手势识别", True, (200, 200, 200))
        self.screen.blit(title_surface, (camera_x, camera_y - 20))
        
        # 在摄像头窗口下方显示阵型相关信息
        recognized_formation = self.recognize_formation()
        best_match = self.find_closest_standard_formation(recognized_formation)
        best_match_text = best_match if best_match else "无"
        
        info_y = camera_y + camera_height + 8
        info_y = self.draw_camera_info_line(f"识别阵型: {recognized_formation}", camera_x, info_y)
        info_y = self.draw_camera_info_line(f"匹配阵型: {best_match_text}", camera_x, info_y)
        
        # 显示OK手势保持进度
        if self.hand_detected and self.hand_keypoints:
            if self.hand_keypoints.get('is_ok_gesture', False):
                if self.ok_confirmed_wait_release:
                    ok_text = "OK确认: 已触发(松开可再次确认)"
                else:
                    ok_text = f"OK保持: {min(self.ok_hold_elapsed, self.ok_hold_duration):.1f}/{self.ok_hold_duration:.1f}s"
            else:
                ok_text = f"OK保持: 0.0/{self.ok_hold_duration:.1f}s"
        else:
            ok_text = f"OK保持: 0.0/{self.ok_hold_duration:.1f}s"
        info_y = self.draw_camera_info_line(ok_text, camera_x, info_y)
        
        # 显示阵型调整过程与新阵型介绍（保持显示直到下次调整）
        if self.last_adjustment_to:
            status_text = "调整中" if self.smooth_movement_active else "已调整"
            from_name = self.last_adjustment_from or "未知"
            to_name = self.last_adjustment_to
            info_y = self.draw_camera_info_line(
                f"{status_text}: {from_name} → {to_name}",
                camera_x,
                info_y,
                color=(120, 240, 120),
            )
            
            intro_lines = self.get_formation_intro_lines(to_name)
            if intro_lines:
                for idx, line in enumerate(intro_lines):
                    prefix = "介绍: " if idx == 0 else "       "
                    info_y = self.draw_camera_info_line(
                        f"{prefix}{line}",
                        camera_x,
                        info_y,
                        color=(200, 200, 200),
                    )
        elif self.confirmation_timer > 0 and self.confirmation_message:
            # 未能匹配标准阵型等提示
            info_y = self.draw_camera_info_line(self.confirmation_message, camera_x, info_y, color=(255, 180, 120))
    
    def recognize_formation(self):
        """根据场上球员布局自动识别阵型（基于横向坐标聚类的算法）"""
        if not self.players:
            return "未知阵型"
        
        # 仅识别A队（避免双队同时在场导致识别失真）
        field_players = [p for p in self.players if p.team == "A" and p.role != "GK"]
        if len(field_players) < 6:  # 至少需要6个场上球员
            return "阵容不完整"
        
        # 获取球场边界
        min_x, min_y, max_x, max_y = self.pitch_bounds
        pitch_width = max_x - min_x
        pitch_height = max_y - min_y
        
        # 获取所有球员的坐标信息
        player_coords = [(p.x, p.y, p) for p in field_players]
        
        # 基于你的算法思路：先按横向坐标聚类确定列数
        def cluster_by_x_coordinate(coords):
            """基于横向坐标差值进行聚类，确定列数（密度感知优化版）"""
            if len(coords) < 2:
                return [len(coords)], []
            
            # 按x坐标排序
            sorted_coords = sorted(coords, key=lambda c: c[0])
            x_positions = [coord[0] for coord in sorted_coords]
            
            # 计算相邻球员间的横向距离
            distances = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
            
            # 密度分析：检测球员分布的紧密程度
            def analyze_density(distances):
                """分析球员分布密度，返回密度等级和建议阈值调整因子"""
                if not distances:
                    return "normal", 1.0, []
                
                min_dist = min(distances)
                avg_dist = sum(distances) / len(distances)
                max_dist = max(distances)
                
                # 更细致的密度分级
                if min_dist < 12 and avg_dist < 20:
                    return "extremely_dense", 0.4, distances  # 极致密
                elif min_dist < 15 and avg_dist < 25:
                    return "very_dense", 0.6, distances      # 极密
                elif min_dist < 25 and avg_dist < 35:
                    return "dense", 0.8, distances           # 密集
                elif max_dist > 150:
                    return "sparse", 1.3, distances          # 稀疏
                else:
                    return "normal", 1.0, distances          # 正常
            
            density_level, density_factor, raw_distances = analyze_density(distances)
            
            # 确定聚类阈值的多种方法（考虑密度因素）
            thresholds = []
            
            # 方法1：基于球场宽度的分区（考虑密度调整）
            base_threshold = (pitch_width / 6) * density_factor
            thresholds.append(base_threshold)
            
            # 方法2：基于相邻距离的统计特征（优化计算，加入密度因子）
            if distances:
                avg_distance = sum(distances) / len(distances)
                max_distance = max(distances)
                min_distance = min(distances)
                
                # 优化的统计学方法：根据密度调整阈值
                if density_level == "extremely_dense":
                    # 极致密集群时，使用非常保守的阈值
                    stat_threshold = min(min_distance * 1.8, avg_distance * 1.1, 25)
                elif density_level == "very_dense":
                    # 极密集聚类时，使用更保守的阈值
                    stat_threshold = min(min_distance * 2.2, avg_distance * 1.2, 35)
                elif density_level == "dense":
                    # 密集聚类时，适度放宽阈值
                    stat_threshold = min(avg_distance * 1.3, max_distance * 0.6, 50)
                else:
                    # 正常或稀疏分布时，使用标准阈值
                    stat_threshold = min(avg_distance * 1.5, max_distance * 0.7, 90)
                
                thresholds.append(stat_threshold)
                
                # 方法3：基于最小距离的倍数（根据密度调整倍数）
                if min_distance > 0:
                    if density_level == "extremely_dense":
                        ratio_threshold = min_distance * 1.5  # 极致密时使用最小倍数
                    elif density_level == "very_dense":
                        ratio_threshold = min_distance * 2.0  # 极密时使用较小倍数
                    elif density_level == "dense":
                        ratio_threshold = min_distance * 3.0  # 密集时使用中等倍数
                    else:
                        ratio_threshold = min_distance * 4.0  # 正常时使用较大倍数
                    thresholds.append(ratio_threshold)
                
                # 方法4：基于球场宽度百分比的动态阈值（考虑密度）
                width_percentage_threshold = (pitch_width * 0.15) * density_factor
                thresholds.append(width_percentage_threshold)
            
            # 增加备用阈值：基于球员总数的预期间距
            expected_spacing = pitch_width / (len(coords) - 1) if len(coords) > 1 else 50
            backup_threshold = max(expected_spacing * 0.8, 20)
            thresholds.append(backup_threshold)
            
            # 综合确定最终阈值（改进的权重平均法）
            if len(thresholds) >= 3:
                # 对阈值进行排序并应用权重
                thresholds.sort()
                # 根据密度等级调整权重分配
                if density_level == "extremely_dense":
                    # 极致密集群时极度偏向最小阈值
                    weights = [0.6, 0.3, 0.1] + [0.0] * (len(thresholds) - 3)
                elif density_level == "very_dense":
                    # 极密集群时偏向较小阈值
                    weights = [0.5, 0.3, 0.2] + [0.0] * (len(thresholds) - 3)
                elif density_level == "dense":
                    # 密集群时均衡考虑
                    weights = [0.33, 0.33, 0.34] + [0.0] * (len(thresholds) - 3)
                else:
                    # 正常或稀疏时偏向中等阈值
                    weights = [0.25, 0.35, 0.4] + [0.0] * (len(thresholds) - 3)
                
                # 加权平均计算最终阈值
                threshold = sum(t * w for t, w in zip(thresholds[:3], weights))
            elif len(thresholds) == 2:
                # 两个阈值时取加权平均
                if density_level in ["extremely_dense", "very_dense"]:
                    threshold = thresholds[0] * 0.7 + thresholds[1] * 0.3
                else:
                    threshold = (thresholds[0] + thresholds[1]) / 2
            elif thresholds:
                threshold = thresholds[0]
            else:
                threshold = 50  # 默认阈值
            
            # 添加额外的安全检查
            min_safe_threshold = 10 if density_level == "extremely_dense" else 15
            threshold = max(threshold, min_safe_threshold)  # 根据密度调整最小阈值
            threshold = min(threshold, pitch_width * 0.3)  # 最大阈值不超过球场宽度30%
            
            # 执行聚类（密度感知优化版：改进孤立点处理）
            columns = []
            current_column = [sorted_coords[0]]
            
            # 记录聚类过程用于调试
            clustering_log = []
            
            for i in range(1, len(sorted_coords)):
                distance = sorted_coords[i][0] - sorted_coords[i-1][0]
                
                if distance <= threshold:
                    # 属于同一列
                    current_column.append(sorted_coords[i])
                    clustering_log.append(f"合并: 距离{distance:.1f} <= 阈值{threshold:.1f}")
                else:
                    # 检查是否为孤立点（改进版）
                    is_isolated = False
                    
                    if len(current_column) == 1 and i < len(sorted_coords) - 1:
                        # 单点情况：检查前后距离
                        next_distance = sorted_coords[i+1][0] - sorted_coords[i][0]
                        prev_distance = distance  # 当前距离就是与前一点的距离
                        
                        # 改进的孤立点判断逻辑
                        if next_distance <= threshold * 1.3:  # 与下一点距离较近
                            if prev_distance > threshold * 2:  # 与上一点距离明显较大
                                # 很可能是孤立点，将其归入下一列
                                current_column.append(sorted_coords[i])
                                clustering_log.append(f"孤立点处理: {prev_distance:.1f}>>{next_distance:.1f}")
                                is_isolated = True
                                continue
                        
                        # 密度感知的孤立点处理
                        if density_level in ["dense", "very_dense", "extremely_dense"]:
                            # 在密集群中，更宽松地处理可能的孤立点
                            adjusted_threshold = threshold * (1.2 if density_level == "extremely_dense" else 1.5)
                            if next_distance <= adjusted_threshold:
                                current_column.append(sorted_coords[i])
                                clustering_log.append(f"密集群孤立点处理: 距离{next_distance:.1f}, 阈值{adjusted_threshold:.1f}")
                                is_isolated = True
                                continue
                    
                    if not is_isolated:
                        # 正常的列分割
                        columns.append(current_column)
                        clustering_log.append(f"新建列: 距离{distance:.1f} > 阈值{threshold:.1f}")
                        current_column = [sorted_coords[i]]
            
            # 添加最后一列
            if current_column:
                columns.append(current_column)
                clustering_log.append("添加最后一列")
            
            # 调试信息输出（可选）
            if len(columns) < 2 or len(columns) > 5:
                print(f"密度分析: {density_level}, 阈值: {threshold:.1f}")
                print(f"聚类日志: {clustering_log[-5:] if len(clustering_log) > 5 else clustering_log}")
            
            # 统计每列的球员数量
            column_player_counts = [len(col) for col in columns]
                    
            # 增加上下文验证：基于足球阵型常识进行后处理优化
            def validate_and_correct_clustering(initial_counts, coords, threshold):
                """验证并修正聚类结果，基于足球阵型结构知识"""
                if len(initial_counts) < 2 or len(initial_counts) > 5:
                    return initial_counts
                        
                total_players = len(coords)
                        
                # 特殊处理：针对四列阵型的合理性检查
                if len(initial_counts) == 4 and sum(initial_counts) == 11:
                    # 4-X-X-1模式检查（很可能是错误的）
                    if initial_counts[-1] == 1:
                        print(f"DEBUG: 检测到可疑的4列阵型 {initial_counts}")
                        print(f"DEBUG: 最后一组只有1人，判定为异常")
                                    
                        # 最终解决方案：强制转换为最可能的标准三列阵型
                        # 基于足球常识，11人阵型最常见的是4-4-3或4-3-4等三列阵型
                        # 将4-X-1-1模式转换为4-(X+2)模式
                                    
                        first_group = initial_counts[0]  # 通常是4（后卫）
                        middle_groups_sum = sum(initial_counts[1:-1])  # 中间几组合并
                        last_group = initial_counts[-1]   # 1人（异常组）
                                    
                        # 重新组合为三列：[first_group, middle_groups_sum + last_group, remaining]
                        # 但要确保总数仍然是11人
                        total_middle_and_last = middle_groups_sum + last_group
                        remaining = 11 - first_group - total_middle_and_last
                                    
                        if remaining > 0:
                            corrected = [first_group, total_middle_and_last, remaining]
                            print(f"DEBUG: 强制修正为标准三列阵型 {corrected}")
                            return corrected
                        else:
                            # 如果计算有问题，返回最常见的4-4-3阵型
                            print(f"DEBUG: 返回默认4-4-3阵型")
                            return [4, 4, 3]
                                
                    # 检查是否为其他不合理的四列模式
                    if initial_counts[0] == 1 or initial_counts[-1] == 1:
                        print(f"DEBUG: 检测到边缘单人组 {initial_counts}，重新聚类")
                        # 使用更宽松的阈值重新聚类
                        return initial_counts  # 暂时不处理，记录问题
                            
                # 特殊处理：当识别为单列时，尝试重新分割
                if len(initial_counts) == 1 and initial_counts[0] == total_players:
                    # 极密集群情况，尝试基于间距模式重新分割
                    x_positions = sorted([coord[0] for coord in coords])
                    distances = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
                                
                    if distances:
                        # 寻找最大的间距作为分割点
                        max_gap_idx = distances.index(max(distances))
                        if max(distances) > threshold * 2:  # 明显的分割点
                            first_part = max_gap_idx + 1
                            second_part = len(x_positions) - first_part
                            return [first_part, second_part]
                        
                # 特殊处理：当识别为单列时，尝试重新分割
                if len(initial_counts) == 1 and initial_counts[0] == total_players:
                    # 极密集群情况，尝试基于间距模式重新分割
                    x_positions = sorted([coord[0] for coord in coords])
                    distances = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
                            
                    if distances:
                        # 寻找最大的间距作为分割点
                        max_gap_idx = distances.index(max(distances))
                        if max(distances) > threshold * 2:  # 明显的分割点
                            first_part = max_gap_idx + 1
                            second_part = len(x_positions) - first_part
                            return [first_part, second_part]
                        
                # 特殊处理：当列数过多时，合并相邻的小列
                if len(initial_counts) > 4:
                    corrected = []
                    i = 0
                    while i < len(initial_counts):
                        if initial_counts[i] <= 2 and i < len(initial_counts) - 1:
                            # 小列与下一列合并
                            merged = initial_counts[i] + initial_counts[i+1]
                            corrected.append(merged)
                            i += 2
                        else:
                            corrected.append(initial_counts[i])
                            i += 1
                    return corrected
                        
                return initial_counts
                    
            # 应用验证和修正
            corrected_counts = validate_and_correct_clustering(column_player_counts, coords, threshold)
                                
            return corrected_counts, columns
        
        # 执行横向坐标聚类
        column_counts, columns = cluster_by_x_coordinate(player_coords)
        
        # 如果列数不合理，尝试纵向聚类作为备选
        if len(column_counts) < 2 or len(column_counts) > 5:
            # 备选方案：基于纵向坐标聚类
            def cluster_by_y_coordinate(coords):
                """基于纵向坐标差值进行聚类"""
                if len(coords) < 2:
                    return [len(coords)], []
                
                # 按y坐标排序
                sorted_coords = sorted(coords, key=lambda c: c[1])
                y_positions = [coord[1] for coord in sorted_coords]
                
                # 计算相邻球员间的纵向距离
                distances = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
                
                # 确定纵向阈值
                if distances:
                    avg_distance = sum(distances) / len(distances)
                    max_distance = max(distances)
                    threshold = min(avg_distance * 1.1, max_distance * 0.5, 60)
                else:
                    threshold = pitch_height / 6
                
                # 执行聚类
                rows = []
                current_row = [sorted_coords[0]]
                
                for i in range(1, len(sorted_coords)):
                    distance = sorted_coords[i][1] - sorted_coords[i-1][1]
                    if distance <= threshold:
                        current_row.append(sorted_coords[i])
                    else:
                        rows.append(current_row)
                        current_row = [sorted_coords[i]]
                
                if current_row:
                    rows.append(current_row)
                
                row_player_counts = [len(row) for row in rows]
                return row_player_counts, rows
            
            row_counts, rows = cluster_by_y_coordinate(player_coords)
            
            # 选择更合理的聚类结果
            if 2 <= len(row_counts) <= 4 and (len(column_counts) < 2 or len(column_counts) > 4):
                # 使用纵向聚类结果
                final_counts = row_counts
                layout_type = "纵向排列"
            else:
                # 使用横向聚类结果
                final_counts = column_counts
                layout_type = "横向排列"
        else:
            final_counts = column_counts
            layout_type = "横向排列"
        
        # 转换为阵型表示
        if len(final_counts) >= 3:
            # 多线阵型
            if len(final_counts) == 3:
                line1, line2, line3 = final_counts
                formation_str = f"{line1}-{line2}-{line3}"
            elif len(final_counts) == 4:
                line1, line2, line3, line4 = final_counts
                formation_str = f"{line1}-{line2}-{line3}-{line4}"
            else:
                # 5线或更多线阵型
                formation_str = "-".join(map(str, final_counts))
                
        elif len(final_counts) == 2:
            # 两线阵型
            line1, line2 = final_counts
            formation_str = f"{line1}-{line2}"
            
        elif len(final_counts) == 1:
            # 单线密集阵型
            formation_str = f"{final_counts[0]}人密集阵型"
        else:
            formation_str = f"{len(field_players)}人阵型"
        
        # 可选：打印调试信息
        # print(f"阵型识别 - {layout_type}: {final_counts} -> {formation_str}")
        
        return formation_str
    
    def find_closest_standard_formation(self, recognized_formation):
        """根据识别的阵型找到最接近的标准阵型（改进版：支持列数变化）"""
        if not recognized_formation or recognized_formation in ["未知阵型", "阵容不完整"]:
            return None
            
        # 直接匹配标准阵型名称
        if recognized_formation in self.formation_names:
            return recognized_formation
            
        # 处理数字格式的阵型（如 "3列阵型 (4-3-2)"）
        import re
        # 提取数字阵型格式
        number_match = re.search(r'(\d+-\d+(?:-\d+)*)', recognized_formation)
        if number_match:
            number_formation = number_match.group(1)
            if number_formation in self.formation_names:
                return number_formation
            
            # 智能阵型匹配：考虑列数变化和结构相似性
            # 注意：避免循环引用和模糊匹配
            formation_mapping = {
                # 三列阵型（经典结构）
                '4-4-2': ['4-4-2', '4-4-2菱形', '4-4-1-1'],
                '4-3-3': ['4-3-3', '4-3-3菱形', '4-3-3（伪九号）'],
                '4-2-3-1': ['4-2-3-1', '4-1-4-1'],  # 移除与其他四列阵型的冲突
                '3-5-2': ['3-5-2', '3-4-1-2'],
                '3-4-3': ['3-4-3'],
                '5-3-2': ['5-3-2', '5-3-2（链式防守）', '5-4-1'],
                
                # 四列阵型（增加中间层）
                '4-1-4-1': ['4-1-4-1'],  # 简化，避免与其他阵型冲突
                '4-2-2-2': ['4-2-2-2'],
                '4-3-2-1': ['4-3-2-1', '4-3-2-1（圣诞树）'],  # 保持精确匹配
                '3-4-1-2': ['3-4-1-2'],  # 简化，避免与3-5-2冲突
                
                # 特殊阵型
                '4-4-2钻石': ['4-4-2钻石'],
                '2-3-5': ['2-3-5', '2-3-5（金字塔）'],
                'WM': ['WM', 'WM（3-2-2-3）', '3-2-2-3'],
                '4-2-4': ['4-2-4'],
                '4-6-0': ['4-6-0', '4-6-0（无锋阵）'],
                '3-3-3-1': ['3-3-3-1']
            }
            
            # 基于战术常识的实用匹配算法
            def tactical_match(input_nums):
                """基于足球战术常识的阵型匹配"""
                total_players = sum(input_nums)
                num_cols = len(input_nums)
                
                # 基于总人数和列数的战术分类
                if total_players == 10:
                    if num_cols == 3:
                        # 三列10人阵型
                        if input_nums == [4, 4, 2]:
                            return '4-4-2'
                        elif input_nums == [4, 3, 3]:
                            return '4-3-3'
                        elif input_nums == [3, 5, 2]:
                            return '3-5-2'
                        elif input_nums == [3, 4, 3]:
                            return '3-4-3'
                    elif num_cols == 4:
                        # 四列10人阵型
                        if input_nums == [4, 1, 4, 1]:
                            return '4-1-4-1'
                        elif input_nums == [4, 2, 2, 2]:
                            return '4-2-2-2'
                        elif input_nums == [3, 4, 1, 2]:
                            return '3-4-1-2'
                
                elif total_players == 11:
                    if num_cols == 3:
                        # 三列11人阵型
                        if input_nums[0] == 4 and input_nums[2] >= 2:  # 4-X-2/3结构
                            if input_nums[1] == 4:
                                return '4-4-2'
                            elif input_nums[1] == 3:
                                return '4-3-3'
                            elif input_nums[1] == 5:
                                return '4-2-3-1'
                        elif input_nums[0] == 5 and input_nums[2] == 2:  # 5-3-2结构
                            return '5-3-2（链式防守）'
                        elif input_nums[0] == 3:  # 3后卫体系
                            if input_nums[1] == 5:
                                return '3-5-2'
                            elif input_nums[1] == 4:
                                return '3-4-3'
                    elif num_cols == 4:
                        # 四列11人阵型
                        if input_nums == [4, 1, 4, 2]:
                            return '4-1-4-1'
                        elif input_nums == [4, 2, 3, 2]:
                            return '4-2-3-1'
                        elif input_nums == [4, 3, 2, 2]:
                            return '4-3-2-1（圣诞树）'
                        elif input_nums == [5, 4, 1, 1]:
                            return '5-4-1'
                        elif input_nums == [3, 4, 1, 3]:
                            return '3-4-1-2'
                
                return None
            
            # 尝试战术匹配
            try:
                input_numbers = [int(x) for x in number_formation.split('-')]
                if input_numbers == [3, 2, 2, 3]:
                    for name in self.formation_names:
                        if "WM" in name or "3-2-2-3" in name:
                            return name
                tactical_result = tactical_match(input_numbers)
                if tactical_result and tactical_result in self.formation_names:
                    return tactical_result
            except ValueError:
                pass
            
            # 降级匹配：基础映射表
            # 优先精确匹配，避免模糊匹配导致的错误
            if number_formation in self.formation_names:
                return number_formation
            
            # 检查是否完全匹配某个标准阵型的变体
            for standard_name, variants in formation_mapping.items():
                if number_formation in variants:
                    # 返回完整标准名称
                    for name in self.formation_names:
                        if name == standard_name or name.startswith(standard_name + '(') or standard_name in name:
                            return name
            
            # 如果还是没有匹配，尝试前缀匹配（但要避免歧义）
            for standard_name, variants in formation_mapping.items():
                if any(v.startswith(number_formation) for v in variants):
                    # 确保这不是其他阵型的前缀
                    is_prefix_conflict = False
                    for other_standard, other_variants in formation_mapping.items():
                        if other_standard != standard_name:
                            if any(other_v.startswith(number_formation) for other_v in other_variants):
                                is_prefix_conflict = True
                                break
                    
                    if not is_prefix_conflict:
                        # 返回完整标准名称
                        for name in self.formation_names:
                            if name.startswith(standard_name) or standard_name in name:
                                return name
        
        return None
    
    def adjust_formation_to_standard(self, target_formation, team_name="A", target_formation_b=None):
        """将指定队伍调整为标准阵型（带平滑移动）"""
        if team_name not in ("A", "B", "both"):
            return False
        if not target_formation or target_formation not in self.formation_names:
            return False

        if team_name == "A":
            old_formation = self.selected_formation_a
            target_map = {"A": target_formation}
            self.selected_formation_a = target_map["A"]
            target_teams = ("A",)
        elif team_name == "B":
            old_formation = self.selected_formation_b
            target_map = {"B": target_formation}
            self.selected_formation_b = target_map["B"]
            target_teams = ("B",)
        else:
            old_formation = f"A:{self.selected_formation_a} | B:{self.selected_formation_b}"
            b_target = target_formation_b if target_formation_b in self.formation_names else target_formation
            target_map = {"A": target_formation, "B": b_target}
            self.selected_formation_a = target_map["A"]
            self.selected_formation_b = target_map["B"]
            target_teams = ("A", "B")
        
        # 保存当前球员位置
        old_players_data = []
        for player in self.players:
            old_players_data.append({
                'label': player.label,
                'role': player.role,
                'team': player.team,
                'x': player.x,
                'y': player.y,
                'selected': player.selected,
                'dragging': player.dragging
            })
        
        # 重新创建标准阵型的球员布局
        self.create_players()

        # 建立球员匹配和设置移动目标
        self.movement_targets = {}
        self.selected_player = None
        self.dragging_player = None
        assigned_new = set()
        assigned_old = set()

        def bind_assignment(new_idx, old_idx):
            new_player = self.players[new_idx]
            old_data = old_players_data[old_idx]
            self.movement_targets[new_idx] = {
                'target_x': new_player.x,
                'target_y': new_player.y,
                'start_x': old_data['x'],
                'start_y': old_data['y']
            }
            new_player.selected = old_data['selected']
            new_player.dragging = old_data['dragging']
            new_player.hovered = False
            if new_player.selected:
                self.selected_player = new_player
            if new_player.dragging:
                self.dragging_player = new_player
            assigned_new.add(new_idx)
            assigned_old.add(old_idx)

        # 优先固定门将匹配（仅目标球队）
        if self.players and old_players_data:
            for current_team in target_teams:
                gk_new_idx = None
                gk_old_idx = None
                for idx, player in enumerate(self.players):
                    if player.team == current_team and player.role == "GK":
                        gk_new_idx = idx
                        break
                for idx, old in enumerate(old_players_data):
                    if old.get("team") == current_team and old.get("role") == "GK":
                        gk_old_idx = idx
                        break
                if gk_new_idx is not None and gk_old_idx is not None:
                    bind_assignment(gk_new_idx, gk_old_idx)

        def solve_team_column_min_cost(team_name):
            """
            按列求最小总位移：
            1) 以当前所有球员坐标作为起点
            2) 按新阵型列划分目标位
            3) 在“列”为统计范围下，求全队最小移动量之和
            """
            current_target_formation = target_map.get(team_name, target_formation)
            target_rows = parse_formation_numbers(current_target_formation)
            if not target_rows:
                return

            new_team_field_indices = [
                i for i, p in enumerate(self.players)
                if p.team == team_name and p.role != "GK"
            ]
            old_team_field_indices = [
                i for i, old in enumerate(old_players_data)
                if old.get("team") == team_name and old.get("role") != "GK"
            ]
            if not new_team_field_indices or not old_team_field_indices:
                return

            # 按新阵型划分“新列目标位”
            new_columns = []
            cursor = 0
            for c in target_rows:
                col = new_team_field_indices[cursor: cursor + c]
                if col:
                    new_columns.append(col)
                cursor += c
            if not new_columns:
                return

            n_old = len(old_team_field_indices)
            n_new = sum(len(col) for col in new_columns)
            if n_old != n_new:
                # 兜底：人数不一致时退回最近邻（极少见）
                remaining_old = set(old_team_field_indices)
                for col in new_columns:
                    for new_idx in col:
                        nx, ny = self.players[new_idx].x, self.players[new_idx].y
                        old_idx = min(
                            remaining_old,
                            key=lambda oi: (nx - old_players_data[oi]['x']) ** 2 + (ny - old_players_data[oi]['y']) ** 2
                        )
                        bind_assignment(new_idx, old_idx)
                        remaining_old.remove(old_idx)
                return

            # 旧球员本地编号(0..n_old-1) -> 全局 old_idx
            local_to_old = list(old_team_field_indices)
            all_local = list(range(n_old))

            # 预计算每列的“子集最小代价 + 具体映射”
            # key: (col_idx, subset_mask) -> (cost, pairs[(new_idx, old_global_idx), ...])
            col_cost_cache = {}
            for col_idx, col_new_indices in enumerate(new_columns):
                k = len(col_new_indices)
                targets = [(self.players[ni].x, self.players[ni].y) for ni in col_new_indices]

                for subset_local in itertools.combinations(all_local, k):
                    subset_mask = 0
                    for li in subset_local:
                        subset_mask |= (1 << li)

                    # 在该列内，求“当前这k名球员 -> 这k个目标位”的最小匹配
                    best_cost = None
                    best_pairs = None
                    for perm in itertools.permutations(range(k), k):
                        total = 0.0
                        pairs = []
                        for src_pos, target_pos in enumerate(perm):
                            local_old_idx = subset_local[src_pos]
                            old_global_idx = local_to_old[local_old_idx]
                            ox = old_players_data[old_global_idx]['x']
                            oy = old_players_data[old_global_idx]['y']
                            tx, ty = targets[target_pos]
                            dist = math.hypot(tx - ox, ty - oy)
                            total += dist
                            pairs.append((col_new_indices[target_pos], old_global_idx))
                        if best_cost is None or total < best_cost:
                            best_cost = total
                            best_pairs = pairs

                    col_cost_cache[(col_idx, subset_mask)] = (best_cost, best_pairs)

            # DP：按列分配子集，最小化全队总位移
            # dp[col_i][used_mask] = cost
            full_mask = (1 << n_old) - 1
            dp = {0: {0: 0.0}}
            prev = {}  # (col_i, used_mask) -> (prev_mask, subset_mask, pairs)

            for col_i, col_new_indices in enumerate(new_columns, start=1):
                k = len(col_new_indices)
                dp[col_i] = {}
                for used_mask, used_cost in dp[col_i - 1].items():
                    remain = [li for li in all_local if not (used_mask & (1 << li))]
                    for subset_local in itertools.combinations(remain, k):
                        subset_mask = 0
                        for li in subset_local:
                            subset_mask |= (1 << li)
                        next_mask = used_mask | subset_mask

                        cache_key = (col_i - 1, subset_mask)
                        if cache_key not in col_cost_cache:
                            continue
                        col_cost, pairs = col_cost_cache[cache_key]
                        cand = used_cost + col_cost
                        if next_mask not in dp[col_i] or cand < dp[col_i][next_mask]:
                            dp[col_i][next_mask] = cand
                            prev[(col_i, next_mask)] = (used_mask, subset_mask, pairs)

            if len(new_columns) not in dp or full_mask not in dp[len(new_columns)]:
                # 理论上不会发生，兜底回最近邻
                remaining_old = set(old_team_field_indices)
                for col in new_columns:
                    for new_idx in col:
                        nx, ny = self.players[new_idx].x, self.players[new_idx].y
                        old_idx = min(
                            remaining_old,
                            key=lambda oi: (nx - old_players_data[oi]['x']) ** 2 + (ny - old_players_data[oi]['y']) ** 2
                        )
                        bind_assignment(new_idx, old_idx)
                        remaining_old.remove(old_idx)
                return

            # 回溯最优映射并绑定
            cur_mask = full_mask
            for col_i in range(len(new_columns), 0, -1):
                key = (col_i, cur_mask)
                if key not in prev:
                    continue
                prev_mask, _subset_mask, pairs = prev[key]
                for new_idx, old_global_idx in pairs:
                    if new_idx not in assigned_new and old_global_idx not in assigned_old:
                        bind_assignment(new_idx, old_global_idx)
                cur_mask = prev_mask

        # 目标队执行“按列最小总位移”匹配
        for current_team in target_teams:
            solve_team_column_min_cost(current_team)
        if team_name == "both":
            print(
                "开始动画调整阵型(1.5秒/按列最小总位移): "
                f"A {old_formation} → {target_map['A']} | B → {target_map['B']}"
            )
        else:
            print(f"开始动画调整阵型(1.5秒/按列最小总位移): {team_name}队 {old_formation} → {target_formation}")

        # 兜底：若存在未匹配位置，保持原地
        for i, new_player in enumerate(self.players):
            if i in assigned_new:
                continue
            self.movement_targets[i] = {
                'target_x': new_player.x,
                'target_y': new_player.y,
                'start_x': new_player.x,
                'start_y': new_player.y
            }
            new_player.hovered = False

        # 初始化动画进度
        self.animation_progress = {i: 0.0 for i in self.movement_targets.keys()}

        # 启动平滑移动
        self.smooth_movement_active = True
        return True
    
    def draw_no_camera_warning(self):
        """绘制无摄像头警告信息"""
        sw, sh = self.screen.get_size()
        
        # 绘制警告背景
        warning_rect = pygame.Rect(sw - 300, sh - 100, 280, 80)
        pygame.draw.rect(self.screen, (60, 60, 60), warning_rect)
        pygame.draw.rect(self.screen, (255, 100, 100), warning_rect, 2)
        
        # 绘制警告文字
        warning_text = "⚠ 摄像头不可用"
        warning_surface = self.font_body.render(warning_text, True, (255, 100, 100))
        self.screen.blit(warning_surface, (sw - 290, sh - 90))
        
        info_text = "仅阵型调整功能可用"
        info_surface = self.font_small.render(info_text, True, (200, 200, 200))
        self.screen.blit(info_surface, (sw - 290, sh - 60))
        
        hint_text = "请检查摄像头连接和权限"
        hint_surface = self.font_small.render(hint_text, True, (180, 180, 180))
        self.screen.blit(hint_surface, (sw - 290, sh - 40))

    def maybe_trigger_team_b_auto_switch(self):
        """B队每10秒自动随机切换阵型（与A队独立）"""
        now = time.time()
        if now - self.last_team_b_auto_switch_time < self.team_b_auto_switch_interval:
            return

        candidates = []
        for name in self.formation_names:
            nums = parse_formation_numbers(name)
            if not nums:
                continue
            if name == self.selected_formation_b:
                continue
            candidates.append(name)
        if not candidates:
            return

        target = random.choice(candidates)
        self.last_team_b_auto_switch_time = now
        should_delay_animation = (
            self.a_position_dirty_pending_ok
            or (self.dragging_player is not None and self.dragging_player.team == "A")
            or self.smooth_movement_active
        )
        if should_delay_animation:
            # 延迟期：继续每10秒更新目标，但不立即播动画
            self.pending_team_b_target_formation = target
            self.confirmation_message = f"B队待执行阵型: {target}（等待A队OK后执行）"
            self.confirmation_timer = 150
            print(f"[B队待执行] 已记录最新目标阵型: {target}")
            return

        self.confirmation_message = f"B队自动切换: {self.selected_formation_b} → {target}"
        self.confirmation_timer = 180
        ok = self.adjust_formation_to_standard(target, team_name="B")
        if ok:
            print(f"[B队自动切换] {target}")
        else:
            print(f"[B队自动切换] 切换失败: {target}")
    
    def show_out_of_bounds_warning(self):
        """显示出界警告（包括预警提示）"""
        if self.dragging_player and self.dragging_player.out_of_bounds:
            self.out_of_bounds_message = "❌ 不可出界！请在球场线内操作"
        else:
            self.out_of_bounds_message = "⚠️ 接近边界！请注意不要出界"
        self.out_of_bounds_timer = 180  # 显示3秒 (60 FPS * 3)
    
    def draw_status_info(self):
        """绘制状态信息"""
        sw, sh = self.screen.get_size()
        
        # 手势状态
        status_text = "手势状态: "
        if not self.hand_detected:
            if self.cap is None:
                status_text += "摄像头不可用"
                color = (255, 100, 100)
            else:
                status_text += f"未检测到手 (丢失:{self.lost_tracking_count}/{self.max_lost_frames})"
                color = (200, 200, 200)
        elif self.is_grabbing:
            # 根据弯曲手指数量显示不同的抓取状态
            curled_count = 0
            if self.hand_keypoints:
                if self.hand_keypoints.get('index_extension', 1) < 0.15: curled_count += 1
                if self.hand_keypoints.get('middle_extension', 1) < 0.15: curled_count += 1
                if self.hand_keypoints.get('ring_extension', 1) < 0.15: curled_count += 1
                if self.hand_keypoints.get('pinky_extension', 1) < 0.15: curled_count += 1
            
            if curled_count >= 4:
                grab_type = "全握拳"
            elif curled_count >= 3:
                grab_type = "握拳"
            else:
                grab_type = "半握"
            
            status_text += f"抓取中 ({grab_type})"
            color = (100, 255, 100)
        else:
            status_text += "悬停"
            color = (255, 255, 100)
        
        status_surface = self.font_small.render(status_text, True, color)
        self.screen.blit(status_surface, (20, 20))
        
        # 手部位置
        if self.hand_detected:
            pos_text = f"手部位置: ({self.hand_x}, {self.hand_y})"
            pos_surface = self.font_small.render(pos_text, True, (200, 200, 200))
            self.screen.blit(pos_surface, (20, 50))
            
            # 显示抓取详情
            if self.hand_keypoints:
                pinch_dist = self.hand_keypoints.get('pinch_distance', 0)
                is_pinching = self.hand_keypoints.get('is_pinching', False)
                is_curled = self.hand_keypoints.get('fingers_curled', False)
                is_ok_gesture = self.hand_keypoints.get('is_ok_gesture', False)
                detail_text = f"捏合: {'是' if is_pinching else '否'} | 弯曲: {'是' if is_curled else '否'} | 距离: {pinch_dist:.3f}"
                detail_surface = self.font_small.render(detail_text, True, (180, 180, 180))
                self.screen.blit(detail_surface, (20, 70))
                
                ok_text = f"OK手势: {'是' if is_ok_gesture else '否'}"
                ok_surface = self.font_small.render(ok_text, True, (180, 180, 180))
                self.screen.blit(ok_surface, (20, 90))

        form_text = f"A队阵型: {self.selected_formation_a} | B队阵型: {self.selected_formation_b}"
        form_surface = self.font_small.render(form_text, True, (200, 220, 200))
        self.screen.blit(form_surface, (20, 112))

        elapsed = time.time() - self.last_team_b_auto_switch_time
        left_sec = max(0.0, self.team_b_auto_switch_interval - elapsed)
        if self.a_position_dirty_pending_ok:
            pending = self.pending_team_b_target_formation or "无"
            auto_text = f"B队自动换阵: 已冻结（待执行={pending}）"
        else:
            auto_text = f"B队自动换阵倒计时: {left_sec:.1f}s"
        auto_surface = self.font_small.render(auto_text, True, (180, 180, 210))
        self.screen.blit(auto_surface, (20, 132))
        
        # 出界警告
        if self.out_of_bounds_timer > 0:
            warning_surface = self.font_body.render(self.out_of_bounds_message, True, (255, 50, 50))
            # 在屏幕中央偏上位置显示
            warning_rect = warning_surface.get_rect(center=(sw//2, sh//2 - 50))
            self.screen.blit(warning_surface, warning_rect)
            self.out_of_bounds_timer -= 1
        
        # 交互提示
        hint_text = "提示: A队手动调整后需OK确认，确认时A/B同步调整"
        hint_surface = self.font_small.render(hint_text, True, (150, 150, 150))
        self.screen.blit(hint_surface, (20, sh - 30))
    
    def handle_events(self):
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            # 移除鼠标点击切换阵型的事件处理
            elif event.type == pygame.VIDEORESIZE:
                # 窗口大小改变时重新创建球员
                self.create_players()
        
        return True
    
    def run(self):
        """主运行循环"""
        running = True
        
        while running:
            # 处理事件
            running = self.handle_events()
            if not running:
                break
            
            # 清屏
            self.screen.fill(BACKGROUND_GRAY)
            
            # 读取摄像头帧
            frame_available = False
            frame = None
            
            if self.cap is not None:
                try:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        frame_available = True
                    else:
                        print("警告: 摄像头读取失败，切换到无摄像头模式")
                        self.cap = None  # 标记摄像头不可用
                except Exception as e:
                    print(f"摄像头读取异常: {e}")
                    self.cap = None
            
            if frame_available and frame is not None:
                # 检测手势
                self.is_grabbing, hand_coords, self.hand_keypoints = self.detect_gesture(frame)
                self.hand_detected = hand_coords is not None
                
                if hand_coords:
                    screen_coords = self.map_hand_to_screen(hand_coords)
                    if screen_coords:
                        self.hand_x, self.hand_y = screen_coords
                
                # 处理OK手势确认
                self.handle_ok_gesture_confirmation()
                
                # 处理手势交互
                self.handle_gesture_interaction()
                
                # 更新平滑移动
                self.update_smooth_movement()
                
                # 在原图上绘制手部骨架
                if self.hand_keypoints:
                    frame = self.draw_hand_skeleton(frame, self.hand_keypoints)
                
                # 将摄像头画面嵌入到pygame窗口右上角
                self.embed_camera_frame(frame)
            else:
                # 无摄像头模式 - 仅处理阵型调整功能
                self.hand_detected = False
                self.is_grabbing = False
                self.hand_keypoints = None
                
                # 仍然处理OK手势确认（可以通过其他方式触发）
                self.handle_ok_gesture_confirmation()
                
                # 更新平滑移动
                self.update_smooth_movement()
                
                # 显示提示信息
                self.draw_no_camera_warning()

            # B队每10秒自动随机切换阵型（独立于A队）
            self.maybe_trigger_team_b_auto_switch()
            
            # 绘制球场区域（占用更多空间）
            pitch_rect = self.draw_pitch_area()
            
            # 绘制球员
            for player in self.players:
                player.draw(self.screen, self.font_small)
            
            # 绘制手部位置指示器
            if self.hand_detected:
                indicator_color = (100, 255, 100) if self.is_grabbing else (255, 255, 100)
                pygame.draw.circle(self.screen, indicator_color, 
                                 (self.hand_x, self.hand_y), 8, 2)
                pygame.draw.circle(self.screen, indicator_color, 
                                 (self.hand_x, self.hand_y), 3)
            
            # 绘制状态信息
            self.draw_status_info()
            
            # 更新显示
            pygame.display.flip()
            self.clock.tick(FPS)
        
        # 清理资源
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

def main():
    """主函数"""
    try:
        app = GestureFormationApp()
        app.run()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
