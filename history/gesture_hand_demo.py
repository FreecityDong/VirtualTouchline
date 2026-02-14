import math
import os
import time
import urllib.request
import ssl

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except Exception as exc:  # pragma: no cover - runtime environment dependent
    raise RuntimeError(
        "MediaPipe 未正确安装或版本不兼容。"
        "请先执行: pip uninstall mediapipe -y && pip install mediapipe"
        "。若仍失败，建议使用 Python 3.10/3.11 的虚拟环境。"
    ) from exc

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

BACKGROUND_GRAY = (40, 40, 40)
PANEL_BG = (28, 28, 28)
TEXT_COLOR = (235, 235, 235)

GROUND_IMAGE_PATHS = ("ground.jpg", "ground.png")

CAPTURE_INDEX = 0
MAX_HANDS = 1
DETECTION_CONF = 0.6
TRACKING_CONF = 0.6

GRAB_THRESHOLD = 0.35
RELEASE_THRESHOLD = 0.45
MOVE_THRESHOLD = 0.02

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

PIL_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/Library/Fonts/Microsoft YaHei.ttf",
]


def load_ground_image():
    for path in GROUND_IMAGE_PATHS:
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return img
    raise FileNotFoundError("ground.jpg or ground.png not found")


def compute_image_rect(area_w, area_h, img_w, img_h, margin):
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


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def ensure_model(path, url):
    if os.path.exists(path):
        return path
    env_path = os.getenv("HAND_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    print("Downloading hand model...")
    try:
        urllib.request.urlretrieve(url, path)
        return path
    except Exception:
        try:
            import certifi

            ctx = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(url, context=ctx) as resp, open(path, "wb") as f:
                f.write(resp.read())
            return path
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            raise RuntimeError(
                "模型下载失败（SSL 证书问题）。"
                "解决方案：\n"
                "1) 安装证书库: pip install certifi\n"
                "2) 手动下载模型并放到项目根目录：hand_landmarker.task\n"
                "   或设置环境变量 HAND_MODEL_PATH 指向模型文件。\n"
                "下载地址见代码中的 MODEL_URL"
            ) from exc


def palm_center(landmarks):
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    return (
        (wrist.x + index_mcp.x + pinky_mcp.x) / 3.0,
        (wrist.y + index_mcp.y + pinky_mcp.y) / 3.0,
    )


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def main():
    ground = load_ground_image()
    ground_h, ground_w = ground.shape[:2]

    cap = cv2.VideoCapture(CAPTURE_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    model_path = ensure_model(MODEL_PATH, MODEL_URL)
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=DETECTION_CONF,
        min_hand_presence_confidence=DETECTION_CONF,
        min_tracking_confidence=TRACKING_CONF,
    )

    grabbed = False
    last_center = None
    last_gesture = "无"
    mirror = True

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            gesture = "无"
            cursor_pos = None

            if results.hand_landmarks and results.handedness:
                # Pick right hand if available
                hand_index = 0
                for i, handedness in enumerate(results.handedness):
                    label = handedness[0].category_name if handedness else ""
                    if label == "Right":
                        hand_index = i
                        break

                landmarks = results.hand_landmarks[hand_index]
                hand_label = results.handedness[hand_index][0].category_name if results.handedness[hand_index] else ""

                # Palm center
                cx, cy = palm_center(landmarks)
                cursor_pos = (cx, cy)

                thumb_tip = (landmarks[4].x, landmarks[4].y)
                index_tip = (landmarks[8].x, landmarks[8].y)
                wrist = (landmarks[0].x, landmarks[0].y)
                middle_mcp = (landmarks[9].x, landmarks[9].y)

                pinch = dist(thumb_tip, index_tip)
                hand_size = max(1e-6, dist(wrist, middle_mcp))
                pinch_norm = pinch / hand_size

                if pinch_norm < GRAB_THRESHOLD:
                    if not grabbed:
                        gesture = "抓取"
                        grabbed = True
                    else:
                        gesture = "移动"
                else:
                    if grabbed and pinch_norm > RELEASE_THRESHOLD:
                        gesture = "放开"
                        grabbed = False
                    else:
                        gesture = "移动" if grabbed else "待机"

                if grabbed and last_center is not None and cursor_pos is not None:
                    moved = dist(last_center, cursor_pos)
                    if moved < MOVE_THRESHOLD:
                        if gesture == "移动":
                            gesture = "抓取"

                last_center = cursor_pos
                last_gesture = gesture

                # Draw hand skeleton on camera preview
                h, w = frame.shape[:2]
                for a, b in HAND_CONNECTIONS:
                    x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
                    x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (80, 200, 255), 2)
                for lm in landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
                draw_text(frame, f"{hand_label}", (10, 30), 0.7, (0, 255, 0), 2)
            else:
                grabbed = False
                last_center = None
                last_gesture = "无"

            canvas = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), BACKGROUND_GRAY, dtype=np.uint8)

            panel_w = max(340, int(WINDOW_WIDTH * 0.28))
            left_w = WINDOW_WIDTH - panel_w
            cv2.rectangle(canvas, (left_w, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), PANEL_BG, -1)

            # Ground image on left (keep aspect)
            margin = 24
            gx, gy, gw, gh = compute_image_rect(left_w, WINDOW_HEIGHT, ground_w, ground_h, margin)
            ground_resized = cv2.resize(ground, (gw, gh), interpolation=cv2.INTER_AREA)
            canvas[gy : gy + gh, gx : gx + gw] = ground_resized

            # Cursor mapping
            if cursor_pos is not None:
                px = int(gx + cursor_pos[0] * gw)
                py = int(gy + cursor_pos[1] * gh)
                px = max(gx, min(gx + gw - 1, px))
                py = max(gy, min(gy + gh - 1, py))
                cv2.circle(canvas, (px, py), 8, (255, 255, 255), -1)
                cv2.circle(canvas, (px, py), 8, (30, 30, 30), 2)

            # Camera preview (right top)
            preview_w = panel_w - 40
            preview_h = int(preview_w * 0.75)
            preview = cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
            px0 = left_w + 20
            py0 = 20
            canvas[py0 : py0 + preview_h, px0 : px0 + preview_w] = preview
            cv2.rectangle(canvas, (px0, py0), (px0 + preview_w, py0 + preview_h), (70, 70, 70), 1)

            # Gesture text (right bottom)
            text_y = WINDOW_HEIGHT - 40
            draw_text(canvas, f"当前动作：{last_gesture}", (left_w + 20, text_y), 0.8, TEXT_COLOR, 2)
            draw_text(canvas, "按 Q 退出 | M 镜像", (left_w + 20, text_y - 26), 0.5, (180, 180, 180), 1)

            cv2.imshow("Virtual Coach - Hand Control", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mirror = not mirror

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
