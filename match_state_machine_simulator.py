import math
import random
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import ttk
from typing import Dict, List, Optional, Tuple


FEATURE_KEYS = ["AW", "DW", "CN", "HS", "PR", "BS", "TR", "BU"]
FEATURE_LABELS = {
    "AW": "进攻宽度",
    "DW": "防守宽度",
    "CN": "中路人数",
    "HS": "肋部控制",
    "PR": "逼抢强度",
    "BS": "防线稳固",
    "TR": "攻防转换",
    "BU": "出球组织",
}
TEAM_CN = {"A": "A队", "B": "B队"}
LANES = ["L", "C", "R"]
LANE_CN = {"L": "左路", "C": "中路", "R": "右路"}
ZONE_CHAIN = ["DZ", "MC", "AT", "BA", "PA"]
ZONE_LEVEL_CN = {"DZ": "后场", "MC": "中场", "AT": "前场", "BA": "禁区前沿", "PA": "禁区"}
TEAM_STATS_KEYS = [
    "shots",
    "shots_on_target",
    "goals",
    "turnovers",
    "counterattacks",
    "freekicks",
    "penalties",
    "corners",
]


FORMATIONS: Dict[str, Dict[str, float]] = {
    "4-4-2": {"AW": 0.65, "DW": 0.70, "CN": 0.55, "HS": 0.45, "PR": 0.55, "BS": 0.70, "TR": 0.70, "BU": 0.55},
    "4-4-1-1": {"AW": 0.62, "DW": 0.72, "CN": 0.60, "HS": 0.50, "PR": 0.55, "BS": 0.72, "TR": 0.65, "BU": 0.60},
    "4-3-3": {"AW": 0.80, "DW": 0.62, "CN": 0.55, "HS": 0.55, "PR": 0.70, "BS": 0.55, "TR": 0.70, "BU": 0.62},
    "4-2-3-1": {"AW": 0.70, "DW": 0.70, "CN": 0.65, "HS": 0.60, "PR": 0.62, "BS": 0.70, "TR": 0.60, "BU": 0.70},
    "4-1-4-1": {"AW": 0.68, "DW": 0.72, "CN": 0.62, "HS": 0.55, "PR": 0.58, "BS": 0.74, "TR": 0.58, "BU": 0.62},
    "4-3-2-1": {"AW": 0.55, "DW": 0.68, "CN": 0.75, "HS": 0.72, "PR": 0.60, "BS": 0.68, "TR": 0.55, "BU": 0.72},
    "4-2-2-2": {"AW": 0.58, "DW": 0.65, "CN": 0.70, "HS": 0.72, "PR": 0.62, "BS": 0.66, "TR": 0.62, "BU": 0.68},
    "4-4-2钻石": {"AW": 0.50, "DW": 0.60, "CN": 0.78, "HS": 0.70, "PR": 0.60, "BS": 0.65, "TR": 0.62, "BU": 0.70},
    "3-5-2": {"AW": 0.70, "DW": 0.65, "CN": 0.70, "HS": 0.60, "PR": 0.60, "BS": 0.68, "TR": 0.62, "BU": 0.65},
    "3-4-3": {"AW": 0.78, "DW": 0.55, "CN": 0.55, "HS": 0.60, "PR": 0.68, "BS": 0.55, "TR": 0.72, "BU": 0.58},
    "3-4-1-2": {"AW": 0.62, "DW": 0.60, "CN": 0.70, "HS": 0.65, "PR": 0.62, "BS": 0.62, "TR": 0.60, "BU": 0.68},
    "5-3-2": {"AW": 0.55, "DW": 0.78, "CN": 0.62, "HS": 0.55, "PR": 0.45, "BS": 0.82, "TR": 0.60, "BU": 0.50},
    "5-4-1": {"AW": 0.45, "DW": 0.82, "CN": 0.58, "HS": 0.48, "PR": 0.40, "BS": 0.85, "TR": 0.55, "BU": 0.45},
    "2-3-5": {"AW": 0.85, "DW": 0.40, "CN": 0.60, "HS": 0.55, "PR": 0.65, "BS": 0.35, "TR": 0.75, "BU": 0.55},
    "WM(3-2-2-3)": {"AW": 0.70, "DW": 0.55, "CN": 0.60, "HS": 0.55, "PR": 0.55, "BS": 0.55, "TR": 0.65, "BU": 0.55},
    "4-2-4": {"AW": 0.82, "DW": 0.50, "CN": 0.55, "HS": 0.55, "PR": 0.60, "BS": 0.50, "TR": 0.75, "BU": 0.52},
    "5-3-2链式防守": {"AW": 0.50, "DW": 0.85, "CN": 0.60, "HS": 0.50, "PR": 0.35, "BS": 0.90, "TR": 0.58, "BU": 0.45},
    "4-6-0": {"AW": 0.60, "DW": 0.70, "CN": 0.80, "HS": 0.75, "PR": 0.70, "BS": 0.60, "TR": 0.55, "BU": 0.78},
    "4-3-3伪九号": {"AW": 0.75, "DW": 0.60, "CN": 0.65, "HS": 0.70, "PR": 0.70, "BS": 0.55, "TR": 0.68, "BU": 0.75},
    "3-3-3-1": {"AW": 0.68, "DW": 0.58, "CN": 0.70, "HS": 0.65, "PR": 0.65, "BS": 0.58, "TR": 0.65, "BU": 0.65},
}

ROUTE_BASE: Dict[str, Tuple[float, float, float]] = {
    "4-4-2": (0.34, 0.32, 0.34),
    "4-4-1-1": (0.32, 0.36, 0.32),
    "4-3-3": (0.40, 0.20, 0.40),
    "4-2-3-1": (0.32, 0.36, 0.32),
    "4-1-4-1": (0.31, 0.38, 0.31),
    "4-3-2-1": (0.24, 0.52, 0.24),
    "4-2-2-2": (0.28, 0.44, 0.28),
    "4-4-2钻石": (0.22, 0.56, 0.22),
    "3-5-2": (0.35, 0.30, 0.35),
    "3-4-3": (0.38, 0.24, 0.38),
    "3-4-1-2": (0.30, 0.40, 0.30),
    "5-3-2": (0.33, 0.34, 0.33),
    "5-4-1": (0.28, 0.44, 0.28),
    "2-3-5": (0.42, 0.16, 0.42),
    "WM(3-2-2-3)": (0.34, 0.32, 0.34),
    "4-2-4": (0.41, 0.18, 0.41),
    "5-3-2链式防守": (0.26, 0.48, 0.26),
    "4-6-0": (0.24, 0.52, 0.24),
    "4-3-3伪九号": (0.35, 0.30, 0.35),
    "3-3-3-1": (0.33, 0.34, 0.33),
}

HEIGHT_STRATEGIES = {
    "默认平衡": {"d": {k: 0.0 for k in FEATURE_KEYS}, "risk": 0.0, "route": (0.0, 0.0, 0.0)},
    "前压": {"d": {"AW": 0.05, "DW": -0.05, "CN": 0.05, "HS": 0.05, "PR": 0.12, "BS": -0.08, "TR": 0.05, "BU": -0.02}, "risk": 0.12, "route": (0.0, 0.0, 0.0)},
    "后撤": {"d": {"AW": -0.05, "DW": 0.08, "CN": 0.02, "HS": -0.02, "PR": -0.10, "BS": 0.12, "TR": -0.05, "BU": -0.02}, "risk": -0.10, "route": (0.0, 0.0, 0.0)},
    "深度防守": {"d": {"AW": -0.05, "DW": 0.10, "CN": 0.03, "HS": -0.03, "PR": -0.18, "BS": 0.15, "TR": -0.05, "BU": -0.05}, "risk": -0.12, "route": (0.0, 0.0, 0.0)},
}
TEMPO_STRATEGIES = {
    "默认平衡": {"d": {k: 0.0 for k in FEATURE_KEYS}, "risk": 0.0, "route": (0.0, 0.0, 0.0)},
    "控球推进": {"d": {"AW": -0.02, "DW": 0.03, "CN": 0.08, "HS": 0.06, "PR": -0.02, "BS": 0.02, "TR": -0.08, "BU": 0.18}, "risk": -0.03, "route": (-0.03, 0.06, -0.03)},
    "快速反击": {"d": {"AW": 0.05, "DW": -0.02, "CN": -0.02, "HS": -0.02, "PR": 0.05, "BS": -0.05, "TR": 0.18, "BU": -0.08}, "risk": 0.06, "route": (0.04, -0.08, 0.04)},
}
CHANNEL_STRATEGIES = {
    "默认平衡": {"d": {k: 0.0 for k in FEATURE_KEYS}, "risk": 0.0, "route": (0.0, 0.0, 0.0)},
    "边路倾斜": {"d": {"AW": 0.18, "DW": 0.02, "CN": -0.05, "HS": -0.08, "PR": 0.02, "BS": -0.02, "TR": 0.05, "BU": -0.02}, "risk": 0.03, "route": (0.12, -0.24, 0.12)},
    "中路渗透": {"d": {"AW": -0.08, "DW": -0.02, "CN": 0.15, "HS": 0.12, "PR": 0.02, "BS": -0.02, "TR": 0.02, "BU": 0.05}, "risk": 0.04, "route": (-0.10, 0.20, -0.10)},
}
SHOT_TYPES = {
    "Long": (0.05, -0.08),
    "Box": (0.08, 0.00),
    "BigChance": (0.10, 0.12),
    "Header": (0.08, -0.02),
    "SetPiece": (0.06, 0.05),
    "Penalty": (0.78, 0.18),
    "Rebound": (0.12, 0.10),
    "LowDrive": (0.09, 0.07),
    "Finesse": (0.10, 0.06),
    "CornerShot": (0.05, -0.04),
    "CurledFK": (0.07, 0.06),
    "KnuckleFK": (0.06, 0.04),
}
SHOT_TYPE_CN = {
    "Long": "远射",
    "Box": "禁区内射门",
    "BigChance": "绝佳机会射门",
    "Header": "头球",
    "SetPiece": "定位球射门",
    "Penalty": "点球",
    "Rebound": "补射",
    "LowDrive": "低平球射门",
    "Finesse": "兜射",
    "CornerShot": "角球直接攻门",
    "CurledFK": "弧线任意球",
    "KnuckleFK": "电梯任意球",
}

ZONE_ATTACK_W = {
    "DZ": {"AW": 0.08, "CN": 0.12, "HS": 0.10, "TR": 0.05, "BU": 0.30},
    "MC": {"AW": 0.16, "CN": 0.18, "HS": 0.18, "TR": 0.10, "BU": 0.20},
    "AT": {"AW": 0.22, "CN": 0.22, "HS": 0.25, "TR": 0.10, "BU": 0.10},
    "BA": {"AW": 0.28, "CN": 0.18, "HS": 0.24, "TR": 0.10, "BU": 0.06},
    "PA": {"AW": 0.10, "CN": 0.35, "HS": 0.30, "TR": 0.12, "BU": 0.05},
}
ZONE_DEF_W = {
    "DZ": {"DW": 0.10, "CN": 0.10, "PR": 0.12, "BS": 0.30, "TR": 0.06},
    "MC": {"DW": 0.16, "CN": 0.14, "PR": 0.20, "BS": 0.18, "TR": 0.12},
    "AT": {"DW": 0.20, "CN": 0.10, "PR": 0.22, "BS": 0.16, "TR": 0.10},
    "BA": {"DW": 0.22, "CN": 0.08, "PR": 0.20, "BS": 0.18, "TR": 0.10},
    "PA": {"DW": 0.08, "CN": 0.10, "PR": 0.24, "BS": 0.24, "TR": 0.10},
}


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def normalize_route(route: Tuple[float, float, float]) -> Tuple[float, float, float]:
    bounded = [clamp(x, 0.05, 0.90) for x in route]
    s = sum(bounded)
    if s <= 0:
        return 0.33, 0.34, 0.33
    return bounded[0] / s, bounded[1] / s, bounded[2] / s


def pick_lane(weights: Tuple[float, float, float]) -> str:
    r = random.random()
    if r < weights[0]:
        return "L"
    if r < weights[0] + weights[1]:
        return "C"
    return "R"


def pick_lane_with_trace(weights: Tuple[float, float, float]) -> Tuple[str, float]:
    r = random.random()
    if r < weights[0]:
        return "L", r
    if r < weights[0] + weights[1]:
        return "C", r
    return "R", r


def pick_lane_excluding_current(weights: Tuple[float, float, float], current_lane: str) -> Tuple[str, float]:
    lane_to_idx = {"L": 0, "C": 1, "R": 2}
    candidates = [l for l in LANES if l != current_lane]
    c_weights = [weights[lane_to_idx[l]] for l in candidates]
    s = c_weights[0] + c_weights[1]
    if s <= 1e-8:
        r = random.random()
        return (candidates[0] if r < 0.5 else candidates[1]), r
    r = random.random()
    threshold = c_weights[0] / s
    return (candidates[0] if r < threshold else candidates[1]), r


def apply_lane_fatigue(
    weights: Tuple[float, float, float],
    current_lane: str,
    lane_streak: int,
    k_lane_fatigue: float = 0.6,
) -> Tuple[Tuple[float, float, float], float]:
    lane_to_idx = {"L": 0, "C": 1, "R": 2}
    idx = lane_to_idx[current_lane]
    mask = [1.0, 1.0, 1.0]
    fatigue_factor = math.exp(-k_lane_fatigue * lane_streak)
    mask[idx] = fatigue_factor
    eff = (weights[0] * mask[0], weights[1] * mask[1], weights[2] * mask[2])
    return normalize_route(eff), fatigue_factor


def zone_tuple_to_str(zone: Tuple[str, str]) -> str:
    return f"{zone[0]}_{zone[1]}"


def zone_tuple_to_cn(zone: Tuple[str, str]) -> str:
    return f"{ZONE_LEVEL_CN.get(zone[0], zone[0])}{LANE_CN.get(zone[1], zone[1])}"


def format_match_time(total_seconds: int) -> str:
    mins = total_seconds // 60
    secs = total_seconds % 60
    return f"{mins:02d}:{secs:02d}"


def make_empty_team_stats() -> Dict[str, int]:
    return {k: 0 for k in TEAM_STATS_KEYS}


def make_empty_match_stats() -> Dict[str, Dict[str, int]]:
    return {"A": make_empty_team_stats(), "B": make_empty_team_stats()}


def shot_type_to_cn(shot_type: str) -> str:
    return SHOT_TYPE_CN.get(shot_type, shot_type)


def zone_advance(zone: Tuple[str, str]) -> Tuple[str, str]:
    level, lane = zone
    idx = ZONE_CHAIN.index(level)
    if idx >= len(ZONE_CHAIN) - 1:
        return zone
    return ZONE_CHAIN[idx + 1], lane


def zone_to_middle(zone: Tuple[str, str]) -> Tuple[str, str]:
    level, _ = zone
    return level, "C"


def zone_switch(zone: Tuple[str, str], target: str) -> Tuple[str, str]:
    return zone[0], target


def mirror_zone_for_turnover(zone: Tuple[str, str]) -> Tuple[str, str]:
    # 球权交换时的区域转换（以可解释和稳定为优先）
    # 关键修正：中场失误后保持在中场，不再直接跳到禁区前沿
    level_map = {"DZ": "AT", "MC": "MC", "AT": "MC", "BA": "AT", "PA": "DZ"}
    lane_map = {"L": "R", "C": "C", "R": "L"}
    level, lane = zone
    return level_map.get(level, level), lane_map.get(lane, lane)


def throwin_level_map(level: str) -> str:
    # 界外球重启层级：尽量保持原层级，禁区相关区域回落到前场层级
    mapping = {"DZ": "DZ", "MC": "MC", "AT": "AT", "BA": "AT", "PA": "AT"}
    return mapping.get(level, "MC")


def is_edge_out_candidate(zone: Tuple[str, str]) -> bool:
    # 仅贴近边缘区域允许触发出界：
    # 1) 左/右通道（边线附近）
    # 2) 禁区前沿/禁区（底线附近）
    level, lane = zone
    return lane in ("L", "R") or level in ("BA", "PA")


@dataclass
class TeamTactics:
    formation: str = "4-4-2"
    strategy_height: str = "默认平衡"
    strategy_tempo: str = "默认平衡"
    strategy_channel: str = "默认平衡"

    def strategy_names(self) -> List[str]:
        names = []
        for item in [self.strategy_height, self.strategy_tempo, self.strategy_channel]:
            if item != "默认平衡":
                names.append(item)
        if not names:
            names.append("默认平衡")
        return names


@dataclass
class TeamWindowQuota:
    formation_left: int = 1
    strategy_left: int = 2


@dataclass
class MatchState:
    match_seconds: int = 0
    score_a: int = 0
    score_b: int = 0
    possession: str = "A"
    zone: Tuple[str, str] = ("MC", "C")
    last_event: str = "开场"
    last_route: str = "中路"
    lane_streak: int = 0
    team_stats: Dict[str, Dict[str, int]] = field(default_factory=make_empty_match_stats)
    current_window: int = 0


@dataclass
class StepReport:
    summary: str
    detail_lines: List[str] = field(default_factory=list)
    probs: Dict[str, float] = field(default_factory=dict)
    shot_info: Dict[str, float] = field(default_factory=dict)


class MatchSimulatorEngine:
    MATCH_DURATION_SECONDS = 90 * 60
    WINDOW_SECONDS = 15 * 60
    STEP_SECONDS = 15
    LANE_FATIGUE_K = 0.6

    def __init__(self) -> None:
        self.state = MatchState()
        self.team_a = TeamTactics()
        self.team_b = TeamTactics()
        self.quota = {"A": TeamWindowQuota(), "B": TeamWindowQuota()}

    def reset(self) -> None:
        self.state = MatchState()
        self.team_a = TeamTactics()
        self.team_b = TeamTactics()
        self.quota = {"A": TeamWindowQuota(), "B": TeamWindowQuota()}

    def team_obj(self, team: str) -> TeamTactics:
        return self.team_a if team == "A" else self.team_b

    def opp_team(self, team: str) -> str:
        return "B" if team == "A" else "A"

    def calc_features_for_config(
        self,
        formation: str,
        strategy_height: str,
        strategy_tempo: str,
        strategy_channel: str,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], float, Tuple[float, float, float]]:
        base = FORMATIONS[formation].copy()
        weighted = base.copy()
        risk = 0.5
        route = list(ROUTE_BASE[formation])

        strategy_pack = [
            HEIGHT_STRATEGIES[strategy_height],
            TEMPO_STRATEGIES[strategy_tempo],
            CHANNEL_STRATEGIES[strategy_channel],
        ]

        for item in strategy_pack:
            for k in FEATURE_KEYS:
                weighted[k] += item["d"].get(k, 0.0)
            risk += item["risk"]
            route[0] += item["route"][0]
            route[1] += item["route"][1]
            route[2] += item["route"][2]

        for k in FEATURE_KEYS:
            weighted[k] = clamp(weighted[k])
        risk = clamp(risk)
        delta = {k: weighted[k] - base[k] for k in FEATURE_KEYS}
        route_tuple = normalize_route((route[0], route[1], route[2]))
        return base, weighted, delta, risk, route_tuple

    def _calc_weighted_features(self, team: str) -> Tuple[Dict[str, float], Dict[str, float], float, Tuple[float, float, float]]:
        obj = self.team_obj(team)
        base, weighted, _delta, risk, route_tuple = self.calc_features_for_config(
            obj.formation,
            obj.strategy_height,
            obj.strategy_tempo,
            obj.strategy_channel,
        )
        return base, weighted, risk, route_tuple

    def get_snapshot(self) -> Dict[str, object]:
        a_base, a_weight, a_risk, a_route = self._calc_weighted_features("A")
        b_base, b_weight, b_risk, b_route = self._calc_weighted_features("B")
        stats_copy = {
            "A": dict(self.state.team_stats["A"]),
            "B": dict(self.state.team_stats["B"]),
        }
        return {
            "match_seconds": self.state.match_seconds,
            "score": (self.state.score_a, self.state.score_b),
            "possession": self.state.possession,
            "zone": self.state.zone,
            "last_event": self.state.last_event,
            "last_route": self.state.last_route,
            "stats": stats_copy,
            "window": self.state.current_window,
            "A": {"base": a_base, "weighted": a_weight, "risk": a_risk, "route": a_route, "tactics": self.team_a, "quota": self.quota["A"]},
            "B": {"base": b_base, "weighted": b_weight, "risk": b_risk, "route": b_route, "tactics": self.team_b, "quota": self.quota["B"]},
        }

    def apply_team_adjustment(
        self,
        team: str,
        selected_formation: str,
        selected_height: str,
        selected_tempo: str,
        selected_channel: str,
    ) -> Tuple[bool, List[str]]:
        obj = self.team_obj(team)
        quota = self.quota[team]
        msgs: List[str] = []
        any_applied = False

        formation_changed = selected_formation != obj.formation
        strategy_changed = (
            selected_height != obj.strategy_height
            or selected_tempo != obj.strategy_tempo
            or selected_channel != obj.strategy_channel
        )

        if formation_changed:
            if quota.formation_left <= 0:
                msgs.append(f"{team}队本时间窗阵型调整次数已用完")
            else:
                obj.formation = selected_formation
                quota.formation_left -= 1
                any_applied = True
                msgs.append(f"{team}队阵型切换为 {selected_formation}（剩余阵型调整 {quota.formation_left}）")

        if strategy_changed:
            if quota.strategy_left <= 0:
                msgs.append(f"{team}队本时间窗策略调整次数已用完")
            else:
                obj.strategy_height = selected_height
                obj.strategy_tempo = selected_tempo
                obj.strategy_channel = selected_channel
                quota.strategy_left -= 1
                any_applied = True
                strategy_text = " + ".join(obj.strategy_names())
                msgs.append(f"{team}队策略切换为 {strategy_text}（剩余策略调整 {quota.strategy_left}）")

        if not formation_changed and not strategy_changed:
            msgs.append(f"{team}队未检测到变更")
            return False, msgs
        return any_applied, msgs

    def _build_process_factors(self, team: str, opp: str, zone_level: str) -> Dict[str, float]:
        _, t, risk_t, _ = self._calc_weighted_features(team)
        _, o, _, _ = self._calc_weighted_features(opp)

        attack_term = sum(ZONE_ATTACK_W[zone_level][k] * t[k] for k in ZONE_ATTACK_W[zone_level])
        defend_term = sum(ZONE_DEF_W[zone_level][k] * o[k] for k in ZONE_DEF_W[zone_level])
        c_attack = sigmoid((attack_term - 0.40) * 5)
        c_defense = sigmoid((defend_term - 0.40) * 5)

        zone_progress_bias = {"DZ": -0.10, "MC": -0.04, "AT": 0.03, "BA": 0.08, "PA": 0.12}[zone_level]
        zone_entry_bias = {"DZ": -0.20, "MC": -0.10, "AT": 0.02, "BA": 0.10, "PA": 0.18}[zone_level]
        zone_disrupt_bias = {"DZ": 0.06, "MC": 0.02, "AT": -0.01, "BA": -0.03, "PA": -0.05}[zone_level]

        f_build = sigmoid(3.0 * (t["BU"] - o["PR"]) + 0.4 * (c_attack - c_defense))
        f_progress = sigmoid(3.0 * ((0.6 * t["AW"] + 0.4 * t["HS"]) - (0.6 * o["DW"] + 0.4 * o["BS"])) + zone_progress_bias)
        f_entry = sigmoid(3.0 * (0.5 * t["CN"] + 0.5 * t["HS"] - o["BS"]) + zone_entry_bias)
        f_transition = sigmoid(3.0 * (t["TR"] + risk_t - o["BS"] - 0.35) + 0.3 * (c_attack - c_defense))
        f_disrupt = sigmoid(3.0 * (o["PR"] + o["BS"] - (t["BU"] + t["TR"])) + zone_disrupt_bias)

        adv1 = t["AW"] - (o["DW"] + o["BS"]) / 2
        adv2 = (t["CN"] + t["BU"]) / 2 - (o["CN"] + o["PR"]) / 2
        adv3 = (t["HS"] + t["BU"]) / 2 - (o["CN"] + o["BS"]) / 2
        adv4 = (risk_t - t["BS"]) - o["TR"]
        adv5 = t["BU"] - o["PR"]
        matchup = clamp(
            0.7 + 0.6 * (0.22 * adv1 + 0.22 * adv2 + 0.20 * adv3 + 0.18 * adv4 + 0.18 * adv5) + 0.20 * (c_attack - c_defense),
            0.4,
            1.4,
        )

        return {
            "c_attack": c_attack,
            "c_defense": c_defense,
            "f_build": f_build,
            "f_progress": f_progress,
            "f_entry": f_entry,
            "f_transition": f_transition,
            "f_disrupt": f_disrupt,
            "matchup": matchup,
            "risk": risk_t,
        }

    def _choose_shot_type(self, zone_level: str, set_piece: str = "") -> str:
        if set_piece == "penalty":
            return "Penalty"
        if set_piece == "corner":
            return "CornerShot"
        if set_piece == "freekick":
            return random.choices(["CurledFK", "KnuckleFK", "SetPiece"], weights=[0.45, 0.25, 0.30], k=1)[0]
        if zone_level == "PA":
            return random.choices(["BigChance", "Box", "Rebound", "LowDrive"], weights=[0.28, 0.40, 0.16, 0.16], k=1)[0]
        if zone_level == "BA":
            return random.choices(["Header", "Box", "Finesse"], weights=[0.35, 0.45, 0.20], k=1)[0]
        if zone_level == "AT":
            return random.choices(["Long", "Finesse", "LowDrive"], weights=[0.45, 0.35, 0.20], k=1)[0]
        return "Long"

    def _resolve_shot(
        self,
        attack_team: str,
        defend_team: str,
        zone: Tuple[str, str],
        process: Dict[str, float],
        set_piece: str = "",
    ) -> Tuple[str, Dict[str, float], List[str]]:
        _, atk, _, _ = self._calc_weighted_features(attack_team)
        _, dfn, _, _ = self._calc_weighted_features(defend_team)
        self._inc_stat(attack_team, "shots")

        shot_type = self._choose_shot_type(zone[0], set_piece)
        base_xg, bonus = SHOT_TYPES[shot_type]
        shot_quality = clamp(base_xg + bonus + 0.28 * (0.6 * process["f_entry"] + 0.4 * process["f_progress"]))

        def_reaction = clamp(0.55 * dfn["PR"] + 0.45 * dfn["BS"])
        gk = clamp(0.5 * dfn["BS"] + 0.5 * dfn["PR"])

        p_goal_raw = sigmoid(2.2 * (shot_quality - 0.45) - 1.0 * def_reaction)
        p_block_raw = sigmoid(1.7 * (0.6 * dfn["PR"] + 0.4 * dfn["BS"] - 0.6 * atk["HS"]))
        p_save_raw = sigmoid(1.8 * (0.7 * dfn["BS"] + 0.6 * gk - 0.5 * shot_quality - 0.30))

        if set_piece == "freekick":
            wall_q = clamp(0.55 * dfn["BS"] + 0.30 * dfn["PR"] + 0.15 * dfn["DW"])
            wall_pos = clamp(0.50 * dfn["DW"] + 0.30 * dfn["PR"] + 0.20 * (0.65 if zone[1] != "C" else 0.50))
            gk_pos = clamp(0.60 * dfn["BS"] + 0.40 * dfn["PR"])
            wall_factor = 0.6 * wall_q + 0.4 * wall_pos
            p_block_raw += 0.2 * wall_factor
            p_save_raw += 0.15 * gk_pos + 0.05 * wall_factor

        s = p_goal_raw + p_block_raw + p_save_raw
        p_goal = p_goal_raw / s
        p_block = p_block_raw / s
        p_save = p_save_raw / s

        r = random.random()
        detail = []
        detail.append(
            f"【随机】射门结果抽样 r_shot={r:.4f}，阈值区间：进球<{p_goal:.4f}，封堵<{(p_goal + p_block):.4f}，其余=扑救"
        )

        if r < p_goal:
            self._inc_stat(attack_team, "shots_on_target")
            self._add_score(attack_team)
            self.state.possession = defend_team
            self.state.zone = ("MC", "C")
            result = "进球"
            detail.append(f"【结果】{TEAM_CN.get(attack_team, attack_team)} {shot_type_to_cn(shot_type)} 破门，比赛回到中圈开球")
        elif r < p_goal + p_block:
            outcome = random.choices(["封堵后角球", "封堵后界外", "封堵后二点争抢"], weights=[0.34, 0.21, 0.45], k=1)[0]
            if outcome == "封堵后角球":
                self.state.zone = ("BA", zone[1])
                self.state.possession = attack_team
                self._inc_stat(attack_team, "corners")
            elif outcome == "封堵后界外":
                self.state.zone = ("AT", zone[1])
                self.state.possession = attack_team
            else:
                self.state.possession = random.choice([attack_team, defend_team])
                if self.state.possession == defend_team:
                    self.state.zone = mirror_zone_for_turnover(("PA", zone[1]))
                    detail.append(
                        f"【结果】二点争抢由{TEAM_CN.get(defend_team, defend_team)}夺得，区域镜像为 {zone_tuple_to_cn(self.state.zone)}"
                    )
                else:
                    self.state.zone = ("PA", zone[1])
            result = outcome
            detail.append(f"【结果】{TEAM_CN.get(defend_team, defend_team)}封堵成功，结果：{outcome}")
        else:
            self._inc_stat(attack_team, "shots_on_target")
            outcome = random.choices(["门将控制", "门将脱手", "门将扑出底线"], weights=[0.55, 0.20, 0.25], k=1)[0]
            if outcome == "门将控制":
                self.state.possession = defend_team
                route_w = self._calc_weighted_features(defend_team)[3]
                lane, r_lane = pick_lane_with_trace(route_w)
                r_lvl = random.random()
                lvl = "MC" if r_lvl < 0.70 else "DZ"
                self.state.zone = (lvl, lane)
                detail.append(
                    "【公式】门将控制回流：先判层级 P(MC)=0.70 / P(DZ)=0.30，"
                    f"再按控球方路线权重分配左右路(左/中/右={route_w[0]:.2f}/{route_w[1]:.2f}/{route_w[2]:.2f})"
                )
                detail.append(
                    f"【随机】门将回流抽样：r_lvl={r_lvl:.4f} -> {ZONE_LEVEL_CN[lvl]}；"
                    f"r_lane={r_lane:.4f} -> {LANE_CN[lane]}"
                )
                detail.append(f"【结果】门将控制后由{TEAM_CN.get(defend_team, defend_team)}在{zone_tuple_to_cn(self.state.zone)}组织")
            elif outcome == "门将脱手":
                self.state.possession = random.choice([attack_team, defend_team])
                if self.state.possession == defend_team:
                    self.state.zone = mirror_zone_for_turnover(("PA", zone[1]))
                    detail.append(f"【结果】脱手后二点由{TEAM_CN.get(defend_team, defend_team)}拿到，区域镜像为 {zone_tuple_to_cn(self.state.zone)}")
                else:
                    self.state.zone = ("PA", zone[1])
            else:
                self.state.possession = attack_team
                self.state.zone = ("BA", zone[1])
                self._inc_stat(attack_team, "corners")
            result = outcome
            if outcome != "门将控制":
                detail.append(f"【结果】{TEAM_CN.get(defend_team, defend_team)}门将处理结果：{outcome}")

        shot_info = {
            "ShotType": shot_type,
            "ShotTypeCN": shot_type_to_cn(shot_type),
            "ShotQuality": shot_quality,
            "P_goal": p_goal,
            "P_block": p_block,
            "P_save": p_save,
        }
        return result, shot_info, detail

    def _add_score(self, team: str) -> None:
        if team == "A":
            self.state.score_a += 1
        else:
            self.state.score_b += 1
        self._inc_stat(team, "goals")

    def _inc_stat(self, team: str, key: str, n: int = 1) -> None:
        if team not in self.state.team_stats:
            return
        if key not in self.state.team_stats[team]:
            return
        self.state.team_stats[team][key] += n

    def _roll_window(self) -> List[str]:
        logs = []
        new_window = min(self.state.match_seconds // self.WINDOW_SECONDS, 5)
        if new_window != self.state.current_window:
            self.state.current_window = new_window
            for t in ["A", "B"]:
                self.quota[t] = TeamWindowQuota()
            logs.append(f"[时间窗刷新] 比赛来到 {format_match_time(self.state.match_seconds)}，两队阵型调整=1次，策略调整=2次")
        return logs

    def step_15min(self) -> StepReport:
        if self.state.match_seconds >= self.MATCH_DURATION_SECONDS:
            return StepReport("比赛已结束", ["比赛结束，停止推进"], {}, {})

        self.state.match_seconds += self.STEP_SECONDS
        logs = self._roll_window()

        attack = self.state.possession
        defend = self.opp_team(attack)
        zone = self.state.zone
        route_raw = self._calc_weighted_features(attack)[3]
        route_eff, fatigue_factor = apply_lane_fatigue(
            route_raw,
            zone[1],
            self.state.lane_streak,
            self.LANE_FATIGUE_K,
        )
        lane = pick_lane(route_eff)
        target_same_lane = lane == zone[1]
        self.state.last_route = LANE_CN[lane]

        process = self._build_process_factors(attack, defend, zone[0])

        shot_rate = (0.55 * (0.4 * process["f_build"] + 0.4 * process["f_progress"] + 0.2 * process["f_transition"]) * (1 - process["f_disrupt"]))
        shot_prob = clamp(1 - math.exp(-shot_rate))

        p_foul = clamp(0.015 + 0.03 * process["risk"] + 0.02 * self._calc_weighted_features(defend)[1]["PR"])
        out_gate = is_edge_out_candidate(zone)
        p_out_base = clamp(0.03 + 0.06 * process["f_disrupt"])
        p_out = p_out_base if out_gate else 0.0

        zone_mul = {"DZ": 0.45, "MC": 0.65, "AT": 1.00, "BA": 1.25, "PA": 1.45}[zone[0]]
        # 射门门控：按区域距离/角度衰减，抑制中后场远射频率
        zone_shot_gate = {"DZ": 0.06, "MC": 0.14, "AT": 0.38, "BA": 0.72, "PA": 1.00}[zone[0]]
        lane_shot_gate = {"L": 0.88, "C": 1.00, "R": 0.88}[zone[1]]
        # 终结区加速：进入 BA/PA 时更倾向尽快完成打门
        shot_endzone_boost = {"DZ": 1.00, "MC": 1.00, "AT": 1.00, "BA": 1.18, "PA": 1.30}[zone[0]]
        p_shot = clamp(shot_prob * zone_mul * zone_shot_gate * lane_shot_gate * shot_endzone_boost)
        p_penalty = p_foul * (0.22 if zone[0] == "PA" else 0.0)
        p_freekick = p_foul * (1.0 - (0.22 if zone[0] == "PA" else 0.0))
        p_event = clamp(p_penalty + p_freekick + p_shot + p_out, 0.0, 0.92)

        route_bias = 0.20 * ({"L": route_eff[0], "C": route_eff[1], "R": route_eff[2]}[lane] - 1.0 / 3.0)
        raw_advance = clamp(sigmoid(2.0 * (process["f_progress"] - 0.45 + route_bias)) - 0.15)
        advance_gate = zone[0] != "PA"
        if not advance_gate:
            raw_advance = 0.0
        switch_endzone_factor = {"DZ": 1.00, "MC": 1.00, "AT": 0.92, "BA": 0.68, "PA": 0.50}[zone[0]]
        raw_switch = clamp(
            sigmoid(
                2.0
                * (
                    0.5 * self._calc_weighted_features(attack)[1]["BU"]
                    + 0.5 * self._calc_weighted_features(attack)[1]["HS"]
                    - 0.5 * self._calc_weighted_features(defend)[1]["PR"]
                    + route_bias
                )
            )
            - 0.20
        ) * switch_endzone_factor
        raw_loss = clamp(
            sigmoid(
                2.0
                * (
                    self._calc_weighted_features(defend)[1]["PR"]
                    + self._calc_weighted_features(defend)[1]["BS"]
                    - (self._calc_weighted_features(attack)[1]["BU"] + self._calc_weighted_features(attack)[1]["TR"])
                )
            )
            - 0.18
        )
        raw_flow_sum = raw_advance + raw_switch + raw_loss
        flow_budget = clamp(1.0 - p_event)
        p_flow = min(flow_budget, 0.55 * raw_flow_sum)
        if raw_flow_sum > 1e-8 and p_flow > 1e-8:
            p_advance = p_flow * raw_advance / raw_flow_sum
            p_switch = p_flow * raw_switch / raw_flow_sum
            p_loss = p_flow * raw_loss / raw_flow_sum
        else:
            p_advance, p_switch, p_loss = 0.0, 0.0, 0.0
        p_hold = clamp(flow_budget - p_flow)
        # 在禁区前沿/禁区减少“原地停留”，将机会倾向回灌到射门触发
        hold_to_shot_ratio = {"DZ": 0.00, "MC": 0.00, "AT": 0.05, "BA": 0.28, "PA": 0.42}[zone[0]]
        if hold_to_shot_ratio > 0:
            hold_to_shot = p_hold * hold_to_shot_ratio
            p_shot_cap = clamp(0.92 - (p_penalty + p_freekick + p_out), 0.0, 0.92)
            p_shot_new = clamp(p_shot + hold_to_shot, 0.0, p_shot_cap)
            shot_add = p_shot_new - p_shot
            p_shot = p_shot_new
            p_hold = clamp(p_hold - shot_add)
            p_event = clamp(p_penalty + p_freekick + p_shot + p_out, 0.0, 0.92)

        probs = {
            "P_penalty": p_penalty,
            "P_freekick": p_freekick,
            "P_shot": p_shot,
            "P_out": p_out,
            "P_event": p_event,
            "P_advance": p_advance,
            "P_switch": p_switch,
            "P_loss": p_loss,
            "P_flow": p_flow,
            "P_hold": p_hold,
            "ShotProb": shot_prob,
            "Matchup": process["matchup"],
            "LaneStreakBefore": float(self.state.lane_streak),
            "TargetSameLane": 1.0 if target_same_lane else 0.0,
        }

        th_pen = p_penalty
        th_fk = th_pen + p_freekick
        th_shot = th_fk + p_shot
        th_out = th_shot + p_out
        th_flow = th_out + p_flow
        r_event = random.random()

        detail = [
            f"【球权】[{format_match_time(self.state.match_seconds)}] 控球方={TEAM_CN.get(attack, attack)}，当前区域={zone_tuple_to_cn(zone)}，路线倾向原始(左/中/右)={route_raw[0]:.2f}/{route_raw[1]:.2f}/{route_raw[2]:.2f}，疲劳后={route_eff[0]:.2f}/{route_eff[1]:.2f}/{route_eff[2]:.2f}，本次目标通道={LANE_CN[lane]}",
            f"【公式】过程因子：组织={process['f_build']:.2f}, 推进={process['f_progress']:.2f}, 入区={process['f_entry']:.2f}, 转换={process['f_transition']:.2f}, 受扰={process['f_disrupt']:.2f}, 对抗修正={process['matchup']:.2f}",
            f"【公式】概率拆分：事件总量={p_event:.2f}(点球={p_penalty:.2f}, 任意球={p_freekick:.2f}, 射门={p_shot:.2f}, 出界={p_out:.2f})；流转总量={p_flow:.2f}(推进={p_advance:.2f}, 转移={p_switch:.2f}, 失误={p_loss:.2f})；停留={p_hold:.2f}",
            f"【公式】射门门控：ZoneMul={zone_mul:.2f}，ZoneShotGate={zone_shot_gate:.2f}，LaneShotGate={lane_shot_gate:.2f}，EndZoneBoost={shot_endzone_boost:.2f}，ShotProb={shot_prob:.2f}",
            f"【公式】终结区修正：SwitchFactor={switch_endzone_factor:.2f}，HoldToShotRatio={hold_to_shot_ratio:.2f}",
            f"【公式】通道疲劳：当前通道={LANE_CN[zone[1]]}，LaneStreak={self.state.lane_streak}，k={self.LANE_FATIGUE_K:.2f}，疲劳因子={fatigue_factor:.3f}",
            f"【公式】出界门控：OutGate(贴边区域)={'是' if out_gate else '否'}，基础出界概率={p_out_base:.2f}，生效后出界概率={p_out:.2f}",
            f"【公式】推进门控：AdvanceGate(非禁区)={'是' if advance_gate else '否'}",
            f"【随机】主抽样值={r_event:.4f}；判定阈值：点球<{th_pen:.4f}，任意球<{th_fk:.4f}，射门<{th_shot:.4f}，出界<{th_out:.4f}，流转<{th_flow:.4f}，其余=停留",
        ]
        shot_info: Dict[str, float] = {}
        resample_hit = False

        if r_event < th_pen:
            detail.append("【判定】命中区间：点球")
            self._inc_stat(attack, "penalties")
            result, shot_info, shot_detail = self._resolve_shot(attack, defend, ("PA", zone[1]), process, set_piece="penalty")
            self.state.last_event = f"点球 -> {result}"
            detail.append("【事件】触发：禁区犯规，判罚点球")
            detail.extend(shot_detail)
        elif r_event < th_fk:
            detail.append("【判定】命中区间：任意球")
            self._inc_stat(attack, "freekicks")
            detail.append("【事件】触发：任意球")
            fk_plan = random.choices(["直接攻门", "传中二点", "短传重启"], weights=[0.36, 0.44, 0.20], k=1)[0]
            detail.append(f"【随机】任意球执行方案={fk_plan}")
            if fk_plan == "直接攻门":
                result, shot_info, shot_detail = self._resolve_shot(attack, defend, zone, process, set_piece="freekick")
                self.state.last_event = f"任意球直接攻门 -> {result}"
                detail.extend(shot_detail)
            elif fk_plan == "传中二点":
                self.state.zone = ("BA", lane)
                self.state.last_event = "任意球传中，形成二点"
                detail.append(f"【结果】任意球传中后落到 {zone_tuple_to_cn(self.state.zone)}")
            else:
                self.state.zone = ("AT", lane)
                self.state.last_event = "任意球短传重启"
                detail.append(f"【结果】任意球短传回做，进入 {zone_tuple_to_cn(self.state.zone)} 组织")
        elif r_event < th_shot:
            detail.append("【判定】命中区间：运动战射门")
            result, shot_info, shot_detail = self._resolve_shot(attack, defend, zone, process)
            self.state.last_event = f"运动战射门 -> {result}"
            detail.append("【事件】触发：运动战射门")
            detail.extend(shot_detail)
        elif r_event < th_out:
            detail.append("【判定】命中区间：出界")
            last_touch_def = random.random() < (0.55 + 0.20 * process["f_disrupt"])
            detail.append(f"【随机】最后触球方判定：防守方触球={str(last_touch_def)}")
            if zone[0] in ["BA", "PA"] and last_touch_def:
                self.state.possession = attack
                self.state.zone = ("BA", lane)
                self.state.last_event = "防守方触球出底线 -> 角球"
                self._inc_stat(attack, "corners")
                detail.append("【事件】触发：角球（防守方解围出底线）")
            else:
                detail.append(
                    "【公式】界外球重启规则：若防守方最后触球，则原进攻方发界外球；"
                    "若进攻方最后触球，则球权交换后按镜像区域发界外球。"
                )
                if last_touch_def:
                    self.state.possession = attack
                    restart_level = throwin_level_map(zone[0])
                    restart_lane = zone[1]
                    self.state.zone = (restart_level, restart_lane)
                    detail.append(
                        f"【结果】界外球重启映射：原区域={zone_tuple_to_cn(zone)}，"
                        f"ThrowInMap层级={ZONE_LEVEL_CN[restart_level]}，通道={LANE_CN[restart_lane]} -> 落点={zone_tuple_to_cn(self.state.zone)}"
                    )
                else:
                    self.state.possession = defend
                    mirrored_zone = mirror_zone_for_turnover(zone)
                    restart_level = throwin_level_map(mirrored_zone[0])
                    self.state.zone = (restart_level, mirrored_zone[1])
                    detail.append(
                        f"【结果】界外球重启映射：先镜像 {zone_tuple_to_cn((zone[0], lane))} -> {zone_tuple_to_cn(mirrored_zone)}，"
                        f"再按ThrowInMap层级={ZONE_LEVEL_CN[restart_level]} -> 落点={zone_tuple_to_cn(self.state.zone)}"
                    )
                self.state.last_event = "出界 -> 界外球重启"
                detail.append("【事件】触发：出界，界外球重启")
        elif r_event < th_flow:
            detail.append("【判定】命中区间：流转事件")
            r_flow = random.random()
            cond_advance = (p_advance / p_flow) if p_flow > 1e-8 else 0.0
            cond_switch = (p_switch / p_flow) if p_flow > 1e-8 else 0.0
            cond_loss = (p_loss / p_flow) if p_flow > 1e-8 else 0.0
            detail.append(
                f"【随机】流转抽样值={r_flow:.4f}；条件阈值：推进<{cond_advance:.4f}，转移<{(cond_advance + cond_switch):.4f}，失误<{(cond_advance + cond_switch + cond_loss):.4f}"
            )
            if r_flow < cond_advance:
                new_zone = zone_advance(zone)
                self.state.zone = zone_switch(new_zone, lane)
                self.state.last_event = f"推进成功 -> {zone_tuple_to_cn(self.state.zone)}"
                detail.append("【结果】流转事件：纵向推进成功")
            elif r_flow < cond_advance + cond_switch:
                switch_lane = lane
                if switch_lane == zone[1]:
                    switch_lane, r_switch = pick_lane_excluding_current(route_eff, zone[1])
                    resample_hit = True
                    detail.append(
                        f"【规则】同路目标会导致无效横传，已重采样通道：r_switch={r_switch:.4f} -> {LANE_CN[switch_lane]}"
                    )
                self.state.zone = zone_switch(zone, switch_lane)
                self.state.last_event = f"横向转移 -> {zone_tuple_to_cn(self.state.zone)}"
                detail.append("【结果】流转事件：同层横向转移")
            elif r_flow < cond_advance + cond_switch + cond_loss:
                self._inc_stat(attack, "turnovers")
                self._inc_stat(defend, "counterattacks")
                self.state.possession = defend
                mirrored = mirror_zone_for_turnover(zone)
                self.state.zone = mirrored
                self.state.last_event = "失误被断，原区反击"
                detail.append(f"【结果】流转事件：失误，球权交换并就地反击（区域镜像：{zone_tuple_to_cn(zone)} -> {zone_tuple_to_cn(mirrored)}）")
            else:
                self.state.last_event = "节奏停留（原地控球）"
                detail.append("【结果】流转事件：无有效推进动作，保持控球")
        else:
            detail.append("【判定】命中区间：停留")
            self.state.last_event = "节奏停留（原地控球）"
            detail.append("【结果】保持当前球权与区域，不发生推进")

        if self.state.possession != attack:
            self.state.lane_streak = 0
        elif target_same_lane:
            self.state.lane_streak = min(self.state.lane_streak + 1, 15)
        else:
            self.state.lane_streak = 0
        probs["LaneStreakAfter"] = float(self.state.lane_streak)
        probs["ResampleHit"] = 1.0 if resample_hit else 0.0
        detail.append(
            f"【规则】通道状态更新：目标是否同路={'是' if target_same_lane else '否'}，重采样命中={'是' if resample_hit else '否'}，LaneStreak -> {self.state.lane_streak}"
        )
        detail.append(f"【球权】下一状态：控球方={TEAM_CN.get(self.state.possession, self.state.possession)}，区域={zone_tuple_to_cn(self.state.zone)}")
        logs.extend(detail)
        summary = f"{format_match_time(self.state.match_seconds)} {self.state.last_event} | 比分 A队 {self.state.score_a}:{self.state.score_b} B队"

        return StepReport(summary=summary, detail_lines=logs, probs=probs, shot_info=shot_info)


class MatchSimulatorUI:
    AUTO_TOTAL_REAL_SECONDS = 5 * 60
    HALF_TIME_SECONDS = 45 * 60
    LOG_FILTER_ITEMS = [
        ("log_ball", "球权"),
        ("log_event", "事件"),
        ("log_result", "结果"),
        ("log_rand", "随机"),
        ("log_formula", "公式"),
        ("log_judge", "判定"),
        ("log_system", "系统"),
        ("log_other", "其他"),
    ]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.engine = MatchSimulatorEngine()
        self.slot_ms = self._compute_auto_slot_ms()
        self.auto_job = None
        self.halftime_auto_paused = False
        self.last_report: StepReport = None
        self.log_records: List[Tuple[str, str]] = []
        self.log_filter_vars: Dict[str, tk.BooleanVar] = {}
        self._syncing_controls = False
        self.form_buttons: Dict[str, List[ttk.Radiobutton]] = {"A": [], "B": []}
        self.strategy_buttons: Dict[str, Dict[str, List[ttk.Radiobutton]]] = {
            "A": {"height": [], "tempo": [], "channel": []},
            "B": {"height": [], "tempo": [], "channel": []},
        }
        self.apply_buttons: Dict[str, ttk.Button] = {}
        self.feature_cells: Dict[str, Dict[str, tk.Label]] = {}
        self.zone_cells: Dict[str, tk.Label] = {}
        self.zone_cell_base_bg: Dict[str, str] = {}

        self.root.title("足球状态机模拟器（阵型/策略可视化）")
        self.root.geometry("1680x980")
        self.root.minsize(1420, 820)

        self._build_layout()
        self._init_control_vars()
        self._render_full_snapshot()
        self._append_log("系统初始化完成：默认双方 4-4-2 + 默认平衡")
        self._append_log(
            f"[系统] 自动推进速度：每{self._fmt_seconds(self.slot_ms / 1000)}秒=赛时{self.engine.STEP_SECONDS}秒，整场约{self.AUTO_TOTAL_REAL_SECONDS / 60:.1f}分钟"
        )

    @staticmethod
    def _fmt_seconds(sec: float) -> str:
        return f"{sec:.2f}".rstrip("0").rstrip(".")

    def _compute_auto_slot_ms(self) -> int:
        total_steps = max(1, math.ceil(self.engine.MATCH_DURATION_SECONDS / self.engine.STEP_SECONDS))
        slot = int(round(self.AUTO_TOTAL_REAL_SECONDS * 1000 / total_steps))
        return max(100, slot)

    def _auto_idle_button_text(self) -> str:
        return f"开始自动推进(每{self._fmt_seconds(self.slot_ms / 1000)}秒={self.engine.STEP_SECONDS}秒赛时)"

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=4)
        self.root.columnconfigure(1, weight=3)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(2, weight=1)
        left.columnconfigure(0, weight=1)

        right_outer = ttk.Frame(self.root, padding=8)
        right_outer.grid(row=0, column=1, sticky="nsew")
        right_outer.rowconfigure(0, weight=1)
        right_outer.columnconfigure(0, weight=1)

        self.right_canvas = tk.Canvas(right_outer, highlightthickness=0)
        self.right_canvas.grid(row=0, column=0, sticky="nsew")
        right_scroll = ttk.Scrollbar(right_outer, orient="vertical", command=self.right_canvas.yview)
        right_scroll.grid(row=0, column=1, sticky="ns")
        self.right_canvas.configure(yscrollcommand=right_scroll.set)

        right = ttk.Frame(self.right_canvas)
        right_window_id = self.right_canvas.create_window((0, 0), window=right, anchor="nw")

        def _on_right_inner_configure(_: tk.Event) -> None:
            self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))

        def _on_right_canvas_configure(event: tk.Event) -> None:
            self.right_canvas.itemconfigure(right_window_id, width=event.width)

        def _on_right_mousewheel(event: tk.Event) -> None:
            if hasattr(event, "delta") and event.delta:
                step = -1 if event.delta > 0 else 1
            elif getattr(event, "num", None) == 4:
                step = -1
            else:
                step = 1
            self.right_canvas.yview_scroll(step, "units")

        right.bind("<Configure>", _on_right_inner_configure)
        self.right_canvas.bind("<Configure>", _on_right_canvas_configure)
        self.right_canvas.bind("<MouseWheel>", _on_right_mousewheel)
        self.right_canvas.bind("<Button-4>", _on_right_mousewheel)
        self.right_canvas.bind("<Button-5>", _on_right_mousewheel)
        right.bind("<MouseWheel>", _on_right_mousewheel)
        right.bind("<Button-4>", _on_right_mousewheel)
        right.bind("<Button-5>", _on_right_mousewheel)

        right.rowconfigure(5, weight=1)
        right.columnconfigure(0, weight=1)

        top_btn = ttk.Frame(left)
        top_btn.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.btn_start = ttk.Button(top_btn, text=self._auto_idle_button_text(), command=self.toggle_auto)
        self.btn_start.pack(side="left", padx=(0, 6))
        ttk.Button(top_btn, text=f"手动推进{self.engine.STEP_SECONDS}秒赛时", command=self.manual_step).pack(side="left", padx=(0, 6))
        ttk.Button(top_btn, text="重置比赛", command=self.reset_match).pack(side="left")

        filter_box = ttk.LabelFrame(left, text="日志筛选", padding=6)
        filter_box.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self._build_log_filters(filter_box)

        self.log_text = tk.Text(left, wrap="word", font=("PingFang SC", 12))
        self.log_text.grid(row=2, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")
        self.log_text.tag_configure("log_ball", foreground="#0d47a1", font=("PingFang SC", 12, "bold"))
        self.log_text.tag_configure("log_event", foreground="#6a1b9a", font=("PingFang SC", 12, "bold"))
        self.log_text.tag_configure("log_result", foreground="#b71c1c", font=("PingFang SC", 12, "bold"))
        self.log_text.tag_configure("log_rand", foreground="#1565c0")
        self.log_text.tag_configure("log_formula", foreground="#1b5e20")
        self.log_text.tag_configure("log_judge", foreground="#ef6c00", font=("PingFang SC", 12, "bold"))
        self.log_text.tag_configure("log_system", foreground="#37474f")
        log_scroll = ttk.Scrollbar(left, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=2, column=1, sticky="ns")
        self.log_text["yscrollcommand"] = log_scroll.set

        info_frame = ttk.LabelFrame(right, text="比赛看板", padding=8)
        info_frame.grid(row=0, column=0, sticky="ew")
        info_frame.columnconfigure(0, weight=1)

        self.lbl_score = ttk.Label(info_frame, text="比分 A 0:0 B", font=("PingFang SC", 18, "bold"))
        self.lbl_score.grid(row=0, column=0, sticky="w")

        self.lbl_meta = ttk.Label(info_frame, text="", font=("PingFang SC", 12))
        self.lbl_meta.grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.lbl_status = ttk.Label(info_frame, text="", font=("PingFang SC", 11))
        self.lbl_status.grid(row=2, column=0, sticky="w", pady=(2, 0))

        zone_frame = ttk.LabelFrame(info_frame, text="区域球权图（7x3）", padding=6)
        zone_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self._build_zone_board(zone_frame)
        self.lbl_zone_hint = ttk.Label(info_frame, text="", font=("PingFang SC", 10))
        self.lbl_zone_hint.grid(row=4, column=0, sticky="w", pady=(4, 0))

        stats_frame = ttk.LabelFrame(right, text="专业统计（实时）", padding=8)
        stats_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        stats_frame.columnconfigure(0, weight=1)
        self.lbl_stats = ttk.Label(stats_frame, text="", font=("Menlo", 10), justify="left")
        self.lbl_stats.grid(row=0, column=0, sticky="w")

        quota_frame = ttk.LabelFrame(right, text="时间窗调整限制", padding=8)
        quota_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        quota_frame.columnconfigure(0, weight=1)
        self.lbl_quota_a = ttk.Label(quota_frame, text="A队 阵型1 策略2", font=("PingFang SC", 11))
        self.lbl_quota_a.grid(row=0, column=0, sticky="w")
        self.lbl_quota_b = ttk.Label(quota_frame, text="B队 阵型1 策略2", font=("PingFang SC", 11))
        self.lbl_quota_b.grid(row=1, column=0, sticky="w")

        probs_frame = ttk.LabelFrame(right, text="关键概率/事件", padding=8)
        probs_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        probs_frame.columnconfigure(0, weight=1)
        self.lbl_probs = ttk.Label(probs_frame, text="", font=("PingFang SC", 10), justify="left")
        self.lbl_probs.grid(row=0, column=0, sticky="w")

        feature_frame = ttk.LabelFrame(right, text="8维评分：基础值 vs 加权值", padding=8)
        feature_frame.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        self._build_feature_table(feature_frame)

        controls = ttk.Notebook(right)
        controls.grid(row=5, column=0, sticky="nsew", pady=(8, 0))

        self.tab_a = self._add_scrollable_tab(controls, "A队战术面板")
        self.tab_b = self._add_scrollable_tab(controls, "B队战术面板")

        self._build_team_controls(self.tab_a, "A")
        self._build_team_controls(self.tab_b, "B")

    def _build_feature_table(self, parent: ttk.LabelFrame) -> None:
        table = tk.Frame(parent, bg="#e5e5e5")
        table.pack(fill="x", expand=False)

        headers = ["维度", "A基础", "A加权", "A变化", "B基础", "B加权", "B变化"]
        widths = [10, 8, 8, 8, 8, 8, 8]
        for c, text in enumerate(headers):
            lbl = tk.Label(
                table,
                text=text,
                bg="#d8d8d8",
                fg="#222222",
                font=("PingFang SC", 11, "bold"),
                padx=6,
                pady=6,
                width=widths[c],
                relief="ridge",
                borderwidth=1,
            )
            lbl.grid(row=0, column=c, sticky="nsew")
            table.columnconfigure(c, weight=1)

        for r, key in enumerate(FEATURE_KEYS, start=1):
            row_bg = "#f6f6f6" if r % 2 else "#f0f0f0"
            self.feature_cells[key] = {}
            values = [
                ("dim", FEATURE_LABELS[key]),
                ("a_base", "0.00"),
                ("a_weight", "0.00"),
                ("a_delta", ""),
                ("b_base", "0.00"),
                ("b_weight", "0.00"),
                ("b_delta", ""),
            ]
            for c, (name, text) in enumerate(values):
                anchor = "w" if name == "dim" else "center"
                lbl = tk.Label(
                    table,
                    text=text,
                    bg=row_bg,
                    fg="#202020",
                    font=("PingFang SC", 12),
                    padx=6,
                    pady=4,
                    width=widths[c],
                    relief="ridge",
                    borderwidth=1,
                    anchor=anchor,
                )
                lbl.grid(row=r, column=c, sticky="nsew")
                self.feature_cells[key][name] = lbl

    def _build_zone_board(self, parent: ttk.LabelFrame) -> None:
        board = tk.Frame(parent, bg="#ececec")
        board.pack(fill="x", expand=False)
        lanes = [("L", "左路"), ("C", "中路"), ("R", "右路")]
        # 绝对球场坐标（A队球门 -> B队球门），用于可视化双方换边后的统一位置
        levels = [
            ("A_PA", "A禁区"),
            ("A_BA", "A禁区前沿"),
            ("A_DZ", "A半场"),
            ("MC", "中场"),
            ("B_DZ", "B半场"),
            ("B_BA", "B禁区前沿"),
            ("B_PA", "B禁区"),
        ]

        tk.Label(board, text="", bg="#e0e0e0", fg="#222", width=8, font=("PingFang SC", 10, "bold")).grid(
            row=0, column=0, sticky="nsew", padx=1, pady=1
        )
        for c, (_, lane_cn) in enumerate(lanes, start=1):
            tk.Label(board, text=lane_cn, bg="#d8d8d8", fg="#222", width=12, font=("PingFang SC", 10, "bold")).grid(
                row=0, column=c, sticky="nsew", padx=1, pady=1
            )

        for r, (level, level_cn) in enumerate(levels, start=1):
            tk.Label(board, text=level_cn, bg="#d8d8d8", fg="#222", width=8, font=("PingFang SC", 10, "bold")).grid(
                row=r, column=0, sticky="nsew", padx=1, pady=1
            )
            for c, (lane, _) in enumerate(lanes, start=1):
                key = f"{level}_{lane}"
                base_bg = "#f3f3f3" if (r + c) % 2 else "#ececec"
                cell = tk.Label(
                    board,
                    text="",
                    bg=base_bg,
                    fg="#222",
                    width=12,
                    height=2,
                    font=("PingFang SC", 10),
                    relief="ridge",
                    borderwidth=1,
                )
                cell.grid(row=r, column=c, sticky="nsew", padx=1, pady=1)
                self.zone_cells[key] = cell
                self.zone_cell_base_bg[key] = base_bg

        for i in range(4):
            board.columnconfigure(i, weight=1)
        for i in range(8):
            board.rowconfigure(i, weight=1)

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

    def _render_zone_board(self, zone: Tuple[str, str], possession: str) -> None:
        for key, cell in self.zone_cells.items():
            cell.configure(text="", bg=self.zone_cell_base_bg.get(key, "#f3f3f3"), fg="#222")

        active_key = self._to_absolute_zone_key(zone, possession)
        if active_key is None:
            self.lbl_zone_hint.configure(text=f"当前球位置：{zone_tuple_to_cn(zone)}  |  控球方：{TEAM_CN.get(possession, possession)}")
            return
        active = self.zone_cells.get(active_key)
        if active is None:
            return

        if possession == "A":
            active_bg = "#ffe3e3"
            active_fg = "#7f1d1d"
        else:
            active_bg = "#e3efff"
            active_fg = "#0f2a6f"
        active.configure(text=f"球权:{TEAM_CN.get(possession, possession)}", bg=active_bg, fg=active_fg)
        self.lbl_zone_hint.configure(
            text=f"当前球位置：{zone_tuple_to_cn(zone)}（映射绝对区域 {active_key.replace('_', '-')})  |  控球方：{TEAM_CN.get(possession, possession)}"
        )

    def _add_scrollable_tab(self, notebook: ttk.Notebook, title: str) -> ttk.Frame:
        outer = ttk.Frame(notebook)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(outer, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        inner = ttk.Frame(canvas, padding=4)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        def _on_mousewheel(event: tk.Event) -> None:
            if hasattr(event, "delta") and event.delta:
                step = -1 if event.delta > 0 else 1
            elif getattr(event, "num", None) == 4:
                step = -1
            else:
                step = 1
            canvas.yview_scroll(step, "units")

        def _bind_mouse(_: tk.Event) -> None:
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel)
            canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_mouse(_: tk.Event) -> None:
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.bind("<Enter>", _bind_mouse)
        canvas.bind("<Leave>", _unbind_mouse)

        notebook.add(outer, text=title)
        return inner

    def _build_team_controls(self, parent: ttk.Frame, team: str) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(5, weight=1)

        hdr = ttk.Label(parent, text=f"{team}队当前配置", font=("PingFang SC", 12, "bold"))
        hdr.grid(row=0, column=0, sticky="w", pady=(0, 6))

        formation_box = ttk.LabelFrame(parent, text="阵型（单选）", padding=6)
        formation_box.grid(row=1, column=0, sticky="ew")

        strat_box = ttk.LabelFrame(parent, text="策略（每组单选）", padding=6)
        strat_box.grid(row=2, column=0, sticky="ew", pady=(6, 0))

        route_box = ttk.LabelFrame(parent, text="进攻路线权重（实时）", padding=6)
        route_box.grid(row=3, column=0, sticky="ew", pady=(6, 0))

        apply_box = ttk.Frame(parent)
        apply_box.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        apply_box.columnconfigure(1, weight=1)

        form_var = tk.StringVar(value="4-4-2")
        setattr(self, f"var_form_{team}", form_var)

        cols = 3
        for i, form in enumerate(FORMATIONS.keys()):
            r = i // cols
            c = i % cols
            btn = ttk.Radiobutton(formation_box, text=form, variable=form_var, value=form)
            btn.grid(row=r, column=c, sticky="w", padx=4, pady=1)
            self.form_buttons[team].append(btn)

        for c in range(cols):
            formation_box.columnconfigure(c, weight=1)

        self._build_strategy_group(strat_box, team, "height", "比赛高度组", list(HEIGHT_STRATEGIES.keys()), 0)
        self._build_strategy_group(strat_box, team, "tempo", "节奏组", list(TEMPO_STRATEGIES.keys()), 1)
        self._build_strategy_group(strat_box, team, "channel", "主攻通道组", list(CHANNEL_STRATEGIES.keys()), 2)

        lbl_route = ttk.Label(route_box, text="左/中/右 = 0.33 / 0.34 / 0.33", font=("PingFang SC", 11))
        lbl_route.pack(anchor="w")
        setattr(self, f"lbl_route_{team}", lbl_route)

        btn_apply = ttk.Button(apply_box, text=f"应用{team}队调整", command=lambda t=team: self.apply_adjustments(t))
        btn_apply.grid(row=0, column=0, sticky="w")
        self.apply_buttons[team] = btn_apply
        ttk.Label(apply_box, text="规则：每15分钟阵型1次，策略2次", font=("PingFang SC", 10)).grid(row=0, column=1, sticky="w", padx=8)

        lbl_lock = ttk.Label(apply_box, text="", font=("PingFang SC", 10))
        lbl_lock.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        setattr(self, f"lbl_lock_{team}", lbl_lock)

    def _build_strategy_group(self, parent: ttk.LabelFrame, team: str, key: str, title: str, values: List[str], row: int) -> None:
        box = ttk.LabelFrame(parent, text=title, padding=4)
        box.grid(row=row, column=0, sticky="ew", pady=2)
        box.columnconfigure(0, weight=1)

        var = tk.StringVar(value="默认平衡")
        setattr(self, f"var_{team}_{key}", var)

        for i, item in enumerate(values):
            btn = ttk.Radiobutton(box, text=item, variable=var, value=item)
            btn.grid(row=i // 3, column=i % 3, sticky="w", padx=4)
            self.strategy_buttons[team][key].append(btn)

    def _build_log_filters(self, parent: ttk.LabelFrame) -> None:
        for col in range(8):
            parent.columnconfigure(col, weight=1)
        for i, (tag, text) in enumerate(self.LOG_FILTER_ITEMS):
            var = tk.BooleanVar(value=True)
            self.log_filter_vars[tag] = var
            btn = ttk.Checkbutton(
                parent,
                text=text,
                variable=var,
                command=self._refresh_log_text,
            )
            btn.grid(row=0, column=i, sticky="w", padx=4, pady=2)

    def _init_control_vars(self) -> None:
        for team in ["A", "B"]:
            vars_to_bind = [
                getattr(self, f"var_form_{team}"),
                getattr(self, f"var_{team}_height"),
                getattr(self, f"var_{team}_tempo"),
                getattr(self, f"var_{team}_channel"),
            ]
            for var in vars_to_bind:
                var.trace_add("write", lambda *_args, t=team: self._on_control_change(t))

    def _on_control_change(self, _team: str) -> None:
        if self._syncing_controls:
            return
        self._render_full_snapshot()

    def _current_selection(self, team: str) -> Tuple[str, str, str, str]:
        return (
            getattr(self, f"var_form_{team}").get(),
            getattr(self, f"var_{team}_height").get(),
            getattr(self, f"var_{team}_tempo").get(),
            getattr(self, f"var_{team}_channel").get(),
        )

    def _preview_by_team(self, team: str) -> Dict[str, object]:
        formation, height, tempo, channel = self._current_selection(team)
        base, weighted, delta, risk, route = self.engine.calc_features_for_config(formation, height, tempo, channel)
        return {
            "formation": formation,
            "height": height,
            "tempo": tempo,
            "channel": channel,
            "base": base,
            "weighted": weighted,
            "delta": delta,
            "risk": risk,
            "route": route,
        }

    @staticmethod
    def _format_delta(delta: float) -> Tuple[str, str]:
        if abs(delta) < 1e-6:
            return "", "#666666"
        if delta > 0:
            return f"+{delta:.2f}", "#d22f2f"
        return f"{delta:.2f}", "#1f8f55"

    @staticmethod
    def _format_pct(num: int, den: int) -> str:
        if den <= 0:
            return "0.0%"
        return f"{(100.0 * num / den):.1f}%"

    def _build_stats_text(self, snap: Dict[str, object]) -> str:
        stats = snap.get("stats", {"A": make_empty_team_stats(), "B": make_empty_team_stats()})
        a = stats.get("A", make_empty_team_stats())
        b = stats.get("B", make_empty_team_stats())
        a_shots = int(a.get("shots", 0))
        b_shots = int(b.get("shots", 0))
        a_sot = int(a.get("shots_on_target", 0))
        b_sot = int(b.get("shots_on_target", 0))
        a_goals = int(a.get("goals", 0))
        b_goals = int(b.get("goals", 0))
        line0 = "指标                  A队        B队"
        line1 = f"射门次数              {a_shots:<10d}{b_shots}"
        line2 = f"射正次数              {a_sot:<10d}{b_sot}"
        line3 = f"射正率                {self._format_pct(a_sot, a_shots):<10}{self._format_pct(b_sot, b_shots)}"
        line4 = f"射门成功率(进球/射门)  {self._format_pct(a_goals, a_shots):<10}{self._format_pct(b_goals, b_shots)}"
        line5 = f"失误次数              {int(a.get('turnovers', 0)):<10d}{int(b.get('turnovers', 0))}"
        line6 = f"防守反击次数          {int(a.get('counterattacks', 0)):<10d}{int(b.get('counterattacks', 0))}"
        line7 = f"任意球次数            {int(a.get('freekicks', 0)):<10d}{int(b.get('freekicks', 0))}"
        line8 = f"角球次数              {int(a.get('corners', 0)):<10d}{int(b.get('corners', 0))}"
        line9 = f"点球次数              {int(a.get('penalties', 0)):<10d}{int(b.get('penalties', 0))}"
        return "\n".join([line0, line1, line2, line3, line4, line5, line6, line7, line8, line9])

    def _sync_controls_with_quota(self, snap: Dict[str, object]) -> None:
        self._syncing_controls = True
        try:
            for team in ["A", "B"]:
                quota = snap[team]["quota"]
                tactics = snap[team]["tactics"]
                lock_msgs = []

                formation_locked = quota.formation_left <= 0
                strategy_locked = quota.strategy_left <= 0

                if formation_locked:
                    getattr(self, f"var_form_{team}").set(tactics.formation)
                    lock_msgs.append("阵型已锁定")
                if strategy_locked:
                    getattr(self, f"var_{team}_height").set(tactics.strategy_height)
                    getattr(self, f"var_{team}_tempo").set(tactics.strategy_tempo)
                    getattr(self, f"var_{team}_channel").set(tactics.strategy_channel)
                    lock_msgs.append("策略已锁定")

                for btn in self.form_buttons[team]:
                    btn.configure(state=("disabled" if formation_locked else "normal"))
                for group in ["height", "tempo", "channel"]:
                    for btn in self.strategy_buttons[team][group]:
                        btn.configure(state=("disabled" if strategy_locked else "normal"))

                if formation_locked and strategy_locked:
                    self.apply_buttons[team].configure(state="disabled")
                else:
                    self.apply_buttons[team].configure(state="normal")

                lock_text = "；".join(lock_msgs) if lock_msgs else "当前时间窗可继续调整"
                getattr(self, f"lbl_lock_{team}").configure(text=lock_text)
        finally:
            self._syncing_controls = False

    @staticmethod
    def _resolve_log_tag(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("【球权】"):
            return "log_ball"
        if stripped.startswith("【事件】"):
            return "log_event"
        if stripped.startswith("【结果】"):
            return "log_result"
        if stripped.startswith("【随机】"):
            return "log_rand"
        if stripped.startswith("【公式】"):
            return "log_formula"
        if stripped.startswith("【判定】"):
            return "log_judge"
        if stripped.startswith("[系统]"):
            return "log_system"
        return "log_other"

    def _insert_log_line(self, text: str, tag: str) -> None:
        self.log_text.configure(state="normal")
        if tag in {"log_ball", "log_event", "log_result", "log_rand", "log_formula", "log_judge", "log_system"}:
            self.log_text.insert("end", text + "\n", (tag,))
        else:
            self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _is_log_tag_enabled(self, tag: str) -> bool:
        var = self.log_filter_vars.get(tag)
        return True if var is None else bool(var.get())

    def _refresh_log_text(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        for text, tag in self.log_records:
            if self._is_log_tag_enabled(tag):
                if tag in {"log_ball", "log_event", "log_result", "log_rand", "log_formula", "log_judge", "log_system"}:
                    self.log_text.insert("end", text + "\n", (tag,))
                else:
                    self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        tag = self._resolve_log_tag(text)
        self.log_records.append((text, tag))
        if self._is_log_tag_enabled(tag):
            self._insert_log_line(text, tag)

    def apply_adjustments(self, team: str) -> None:
        ok, msgs = self.engine.apply_team_adjustment(
            team,
            getattr(self, f"var_form_{team}").get(),
            getattr(self, f"var_{team}_height").get(),
            getattr(self, f"var_{team}_tempo").get(),
            getattr(self, f"var_{team}_channel").get(),
        )
        for m in msgs:
            self._append_log(f"[调整] {m}")
        if not ok:
            self._append_log("[调整] 本次未生效")
        self._render_full_snapshot()

    def manual_step(self) -> None:
        report = self.engine.step_15min()
        self.last_report = report
        self._append_log(f"【结果】[推进] {report.summary}")
        for line in report.detail_lines:
            self._append_log("  " + line)
        self._render_full_snapshot(report)

    def toggle_auto(self) -> None:
        if self.auto_job is None:
            self.btn_start.configure(text="暂停自动推进")
            self._append_log(
                f"[系统] 自动推进已启动：每{self._fmt_seconds(self.slot_ms / 1000)}秒推进{self.engine.STEP_SECONDS}秒比赛时间（整场约{self.AUTO_TOTAL_REAL_SECONDS / 60:.1f}分钟）"
            )
            self._auto_tick()
        else:
            self.root.after_cancel(self.auto_job)
            self.auto_job = None
            self.btn_start.configure(text=self._auto_idle_button_text())
            self._append_log("[系统] 自动推进已暂停")

    def _auto_tick(self) -> None:
        before_seconds = self.engine.state.match_seconds
        self.manual_step()
        after_seconds = self.engine.state.match_seconds
        if (
            not self.halftime_auto_paused
            and before_seconds < self.HALF_TIME_SECONDS <= after_seconds
            and after_seconds < self.engine.MATCH_DURATION_SECONDS
        ):
            self.halftime_auto_paused = True
            self.auto_job = None
            self.btn_start.configure(text="继续下半场自动推进")
            self._append_log("[系统] 已到中场(45:00)，自动推进已暂停，请手动确认后开始下半场")
            return
        if self.engine.state.match_seconds >= self.engine.MATCH_DURATION_SECONDS:
            self.auto_job = None
            self.btn_start.configure(text=self._auto_idle_button_text())
            self._append_log("[系统] 比赛已结束，自动推进停止")
            return
        self.auto_job = self.root.after(self.slot_ms, self._auto_tick)

    def reset_match(self) -> None:
        if self.auto_job is not None:
            self.root.after_cancel(self.auto_job)
            self.auto_job = None
            self.btn_start.configure(text=self._auto_idle_button_text())
        self.engine.reset()
        self.halftime_auto_paused = False
        for team in ["A", "B"]:
            getattr(self, f"var_form_{team}").set("4-4-2")
            getattr(self, f"var_{team}_height").set("默认平衡")
            getattr(self, f"var_{team}_tempo").set("默认平衡")
            getattr(self, f"var_{team}_channel").set("默认平衡")

        self.log_records.clear()
        self._refresh_log_text()

        self.last_report = None
        self._append_log("[系统] 已重置：双方恢复 4-4-2 + 默认平衡")
        self._render_full_snapshot()

    def _render_full_snapshot(self, report: StepReport = None) -> None:
        if report is not None:
            self.last_report = report
        report = report if report is not None else self.last_report

        snap = self.engine.get_snapshot()
        match_seconds = snap["match_seconds"]
        score_a, score_b = snap["score"]
        zone = snap["zone"]
        self._sync_controls_with_quota(snap)

        preview_a = self._preview_by_team("A")
        preview_b = self._preview_by_team("B")

        self.lbl_score.configure(text=f"比分 A {score_a}:{score_b} B")
        self.lbl_meta.configure(text=f"比赛时间: {format_match_time(match_seconds)} / 90:00  | 时间窗: {snap['window'] + 1}/6")

        a_tactics_text = " + ".join([x for x in [preview_a["height"], preview_a["tempo"], preview_a["channel"]] if x != "默认平衡"]) or "默认平衡"
        b_tactics_text = " + ".join([x for x in [preview_b["height"], preview_b["tempo"], preview_b["channel"]] if x != "默认平衡"]) or "默认平衡"
        self.lbl_status.configure(
            text=(
                f"当前控球方: {TEAM_CN.get(snap['possession'], snap['possession'])}  | 区域: {zone_tuple_to_cn(zone)}  | 进攻方向: {snap['last_route']}  | 最新事件: {snap['last_event']}\n"
                f"A阵型:{preview_a['formation']} 策略:{a_tactics_text}  || "
                f"B阵型:{preview_b['formation']} 策略:{b_tactics_text}"
            )
        )
        self._render_zone_board(zone, snap["possession"])
        self.lbl_stats.configure(text=self._build_stats_text(snap))

        quota_a = snap["A"]["quota"]
        quota_b = snap["B"]["quota"]
        self.lbl_quota_a.configure(text=f"A队 剩余: 阵型{quota_a.formation_left}次 / 策略{quota_a.strategy_left}次")
        self.lbl_quota_b.configure(text=f"B队 剩余: 阵型{quota_b.formation_left}次 / 策略{quota_b.strategy_left}次")

        for key in FEATURE_KEYS:
            self.feature_cells[key]["a_base"].configure(text=f"{preview_a['base'][key]:.2f}")
            self.feature_cells[key]["a_weight"].configure(text=f"{preview_a['weighted'][key]:.2f}")
            txt_a, col_a = self._format_delta(preview_a["delta"][key])
            self.feature_cells[key]["a_delta"].configure(text=txt_a, fg=col_a)
            self.feature_cells[key]["b_base"].configure(text=f"{preview_b['base'][key]:.2f}")
            self.feature_cells[key]["b_weight"].configure(text=f"{preview_b['weighted'][key]:.2f}")
            txt_b, col_b = self._format_delta(preview_b["delta"][key])
            self.feature_cells[key]["b_delta"].configure(text=txt_b, fg=col_b)

        ar = preview_a["route"]
        br = preview_b["route"]
        getattr(self, "lbl_route_A").configure(text=f"左/中/右 = {ar[0]:.2f} / {ar[1]:.2f} / {ar[2]:.2f}")
        getattr(self, "lbl_route_B").configure(text=f"左/中/右 = {br[0]:.2f} / {br[1]:.2f} / {br[2]:.2f}")

        if report is None:
            self.lbl_probs.configure(text=f"暂无推进数据，点击“手动推进{self.engine.STEP_SECONDS}秒赛时”或启动自动推进")
            return

        probs = report.probs
        shot = report.shot_info

        line1 = (
            f"点球概率={probs.get('P_penalty', 0):.2f}  任意球概率={probs.get('P_freekick', 0):.2f}  "
            f"射门概率={probs.get('P_shot', 0):.2f}  出界概率={probs.get('P_out', 0):.2f}"
        )
        line2 = (
            f"推进概率={probs.get('P_advance', 0):.2f}  转移概率={probs.get('P_switch', 0):.2f}  "
            f"失误概率={probs.get('P_loss', 0):.2f}  停留概率={probs.get('P_hold', 0):.2f}"
        )
        line3 = (
            f"事件总量={probs.get('P_event', 0):.2f}  流转总量={probs.get('P_flow', 0):.2f}  "
            f"时间片射门触发={probs.get('ShotProb', 0):.2f}  对抗修正系数={probs.get('Matchup', 0):.2f}"
        )
        line4 = ""
        if shot:
            line4 = (
                f"射门类型={shot.get('ShotTypeCN', shot.get('ShotType', '-'))}, 质量={shot.get('ShotQuality', 0):.2f}, "
                f"进球={shot.get('P_goal', 0):.2f}, 封堵={shot.get('P_block', 0):.2f}, 扑救={shot.get('P_save', 0):.2f}"
            )
        self.lbl_probs.configure(text="\n".join([line1, line2, line3, line4]))


def main() -> None:
    root = tk.Tk()
    app = MatchSimulatorUI(root)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()
