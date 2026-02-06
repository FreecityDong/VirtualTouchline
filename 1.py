import re
import sys

import pygame


WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700
FPS = 60

BACKGROUND_GRAY = (40, 40, 40)
SIDEBAR_BG = (28, 28, 28)
SIDEBAR_PANEL = (34, 34, 34)
SIDEBAR_HIGHLIGHT = (62, 62, 62)
TEXT_PRIMARY = (235, 235, 235)
TEXT_MUTED = (190, 190, 190)

GROUND_IMAGE_PATHS = ("ground.jpg", "ground.png")

FORMATIONS_MD = "formations_11v11_overview.md"


def load_ground_image():
    for path in GROUND_IMAGE_PATHS:
        try:
            image = pygame.image.load(path)
            return image.convert()
        except FileNotFoundError:
            continue
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


def get_font(size, bold=False):
    candidates = [
        "PingFang SC",
        "PingFang",
        "Songti SC",
        "Heiti SC",
        "STHeiti",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
    ]
    path = None
    for name in candidates:
        path = pygame.font.match_font(name)
        if path:
            break
    font = pygame.font.Font(path, size) if path else pygame.font.SysFont(None, size)
    font.set_bold(bold)
    return font


def parse_formations_md(path):
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


def wrap_text(text, font, max_width):
    if not text:
        return [""]
    lines = []
    current = ""
    for ch in text:
        test = current + ch
        if font.size(test)[0] <= max_width:
            current = test
        else:
            lines.append(current)
            current = ch
    if current:
        lines.append(current)
    return lines


def parse_formation_numbers(name):
    parts = re.findall(r"\d+", name)
    return [int(p) for p in parts]


def roles_for_rows(rows):
    if len(rows) == 3:
        return ["DEF", "MID", "FWD"]
    if len(rows) == 4:
        if rows == [4, 2, 3, 1]:
            return ["DEF", "DM", "AM", "FWD"]
        if rows == [4, 1, 4, 1]:
            return ["DEF", "DM", "MID", "FWD"]
        if rows == [4, 3, 2, 1]:
            return ["DEF", "MID", "AM", "FWD"]
        if rows == [4, 2, 2, 2]:
            return ["DEF", "DM", "AM", "FWD"]
        return ["DEF", "MID", "MID", "FWD"]
    if len(rows) == 5:
        return ["DEF", "DM", "MID", "AM", "FWD"]
    return ["DEF"] + ["MID"] * (len(rows) - 2) + ["FWD"]


def labels_for_role(role, count):
    if role == "DEF":
        mapping = {
            3: ["左中卫", "中卫", "右中卫"],
            4: ["左后卫", "左中卫", "右中卫", "右后卫"],
            5: ["左翼卫", "左中卫", "中卫", "右中卫", "右翼卫"],
        }
    elif role == "DM":
        mapping = {
            1: ["后腰"],
            2: ["左后腰", "右后腰"],
        }
    elif role == "AM":
        mapping = {
            1: ["前腰"],
            2: ["左前腰", "右前腰"],
            3: ["左前腰", "前腰", "右前腰"],
        }
    elif role == "FWD":
        mapping = {
            1: ["中锋"],
            2: ["左前锋", "右前锋"],
            3: ["左边锋", "中锋", "右边锋"],
        }
    else:  # MID
        mapping = {
            1: ["中场"],
            2: ["左中场", "右中场"],
            3: ["左中场", "中场", "右中场"],
            4: ["左边前卫", "左中场", "右中场", "右边前卫"],
            5: ["左边前卫", "左中场", "中场", "右中场", "右边前卫"],
        }
    if count in mapping:
        return mapping[count]
    return [f"球员{i + 1}" for i in range(count)]


def draw_players(surface, field_rect, formation_name, font):
    x0, y0, w, h = field_rect
    x1 = x0 + w

    rows = parse_formation_numbers(formation_name)
    if not rows:
        return
    roles = roles_for_rows(rows)

    margin_x = w * 0.06
    margin_y = h * 0.08
    start_x = x0 + margin_x * 2.0
    end_x = x1 - margin_x * 1.5

    gk_x = x0 + margin_x * 0.7
    gk_y = y0 + h / 2

    player_r = max(6, int(h * 0.014))
    gk_r = player_r + 2

    pygame.draw.circle(surface, (255, 210, 90), (int(gk_x), int(gk_y)), gk_r)
    pygame.draw.circle(surface, (30, 30, 30), (int(gk_x), int(gk_y)), gk_r, 2)
    gk_label = font.render("门将", True, (245, 245, 245))
    gk_label_rect = gk_label.get_rect(center=(int(gk_x), int(gk_y - gk_r - 10)))
    if gk_label_rect.top < y0 + 2:
        gk_label_rect = gk_label.get_rect(center=(int(gk_x), int(gk_y + gk_r + 10)))
    surface.blit(gk_label, gk_label_rect)

    total_lines = len(rows)
    for i, count in enumerate(rows):
        line_x = start_x + (end_x - start_x) * (i + 1) / (total_lines + 1)
        labels = labels_for_role(roles[i] if i < len(roles) else "MID", count)
        for j in range(count):
            y = y0 + margin_y + (h - 2 * margin_y) * (j + 1) / (count + 1)
            pygame.draw.circle(surface, (142, 202, 230), (int(line_x), int(y)), player_r)
            pygame.draw.circle(surface, (30, 30, 30), (int(line_x), int(y)), player_r, 2)
            label = labels[j] if j < len(labels) else f"P{j + 1}"
            text = font.render(label, True, (245, 245, 245))
            text_rect = text.get_rect(center=(int(line_x), int(y - player_r - 10)))
            if text_rect.top < y0 + 2:
                text_rect = text.get_rect(center=(int(line_x), int(y + player_r + 10)))
            surface.blit(text, text_rect)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Virtual Coach - Pitch (Pygame)")
    clock = pygame.time.Clock()

    font_title = get_font(22, bold=True)
    font_body = get_font(18)
    font_small = get_font(16)

    formations = parse_formations_md(FORMATIONS_MD)
    if not formations:
        formations = {
            "4-4-2": [
                "进攻：边路传中 + 双前锋抢点",
                "防守：两条线稳定，覆盖宽度好",
                "缺点：中场人数偏少",
            ],
            "4-3-3": [
                "进攻：边锋 + 中锋组合强",
                "防守：高位压迫强，但身后空当大",
                "缺点：对边后卫体能要求高",
            ],
            "4-2-3-1": [
                "进攻：前腰组织，边中结合",
                "防守：双后腰保护中路",
                "缺点：单前锋易被孤立",
            ],
        }
    formation_names = list(formations.keys())
    selected = formation_names[0]

    ground = load_ground_image()
    ground_size = ground.get_size()
    scaled = None
    scaled_size = None
    radio_rects = []

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                for name, rect in radio_rects:
                    if rect.collidepoint(mx, my):
                        selected = name
                        break

        screen.fill(BACKGROUND_GRAY)
        sw, sh = screen.get_size()

        sidebar_w = max(320, int(sw * 0.28))
        left_w = sw - sidebar_w

        pygame.draw.rect(screen, SIDEBAR_BG, (left_w, 0, sidebar_w, sh))
        pygame.draw.rect(screen, SIDEBAR_PANEL, (left_w + 12, 12, sidebar_w - 24, sh - 24))

        img_margin = 24
        dest = compute_image_rect(left_w, sh, *ground_size, img_margin)
        if scaled_size != (dest[2], dest[3]):
            scaled = pygame.transform.smoothscale(ground, (dest[2], dest[3]))
            scaled_size = (dest[2], dest[3])
        if scaled is not None:
            screen.blit(scaled, (dest[0], dest[1]))
            draw_players(screen, dest, selected, font_small)

        # Sidebar content
        x = left_w + 24
        y = 24
        title = font_title.render("阵型选择", True, TEXT_PRIMARY)
        screen.blit(title, (x, y))
        y += 34

        radio_rects = []
        item_h = 28
        dot_r = 6
        col_gap = 12
        col_w = (sidebar_w - 48 - col_gap) / 2
        rows = (len(formation_names) + 1) // 2
        for idx, name in enumerate(formation_names):
            row = idx // 2
            col = idx % 2
            row_y = y + row * (item_h + 6)
            col_x = x + col * (col_w + col_gap)
            row_rect = pygame.Rect(col_x, row_y, col_w, item_h)
            center = (int(row_rect.x + 10), int(row_rect.y + item_h / 2))
            is_selected = name == selected
            pygame.draw.rect(screen, SIDEBAR_BG, row_rect, border_radius=4)
            pygame.draw.circle(screen, TEXT_MUTED, center, dot_r, 2)
            if is_selected:
                pygame.draw.circle(screen, TEXT_PRIMARY, center, dot_r - 2)
            label = font_body.render(name, True, TEXT_PRIMARY if is_selected else TEXT_MUTED)
            screen.blit(label, (row_rect.x + 22, row_rect.y + 4))
            radio_rects.append((name, row_rect))
        y += rows * (item_h + 6) + 6

        max_text_w = sidebar_w - 64
        line_h = 20
        lines = []
        items = formations.get(selected, [])
        for level, text in items:
            if level == "blank":
                lines.append(("", 0))
                continue
            indent = 1 if level == "sub" else 0
            wrapped = wrap_text(text, font_small, max_text_w - 14 - indent * 14)
            for wline in wrapped:
                lines.append((wline, indent))
            lines.append(("", 0))
        if lines and lines[-1][0] == "":
            lines.pop()

        header_h = 30
        total_h = header_h + len(lines) * line_h + 8
        panel_bottom = sh - 24
        y = max(y + 16, panel_bottom - total_h)

        section = font_title.render("优缺点概览", True, TEXT_PRIMARY)
        screen.blit(section, (x, y))
        y += header_h

        for line, indent in lines:
            if line == "":
                y += line_h // 2
                continue
            prefix = "• " if indent == 0 else "  - "
            render = font_small.render(f"{prefix}{line}", True, TEXT_MUTED)
            screen.blit(render, (x + indent * 14, y))
            y += line_h

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
