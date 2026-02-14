from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent

SRC_DIR = PROJECT_ROOT / "src"
DOCS_DIR = PROJECT_ROOT / "docs"
IMAGES_DIR = PROJECT_ROOT / "images"
PLAYERS_DIR = IMAGES_DIR / "players"
MODELS_DIR = PROJECT_ROOT / "models"

GROUND_IMAGE_PATHS = (
    str(IMAGES_DIR / "ground.jpg"),
    str(IMAGES_DIR / "ground.png"),
)
FORMATIONS_MD_PATH = str(DOCS_DIR / "formations_11v11_classic.md")
MODEL_PATH = str(MODELS_DIR / "hand_landmarker.task")
BALL_ICON_PATH = str(PLAYERS_DIR / "ball.png")
PLAYER_AVATAR_DIR = str(PLAYERS_DIR)

