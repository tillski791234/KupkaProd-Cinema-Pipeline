# config.py — All settings in one place.
# User-specific paths are loaded from user_settings.json (auto-created on first run).

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "user_settings.json")

# --- Defaults (overridden by user_settings.json) ---
_DEFAULTS = {
    "comfyui_root": r"/opt/llm/comfyui",
    "project_output_root": os.path.join(BASE_DIR, "output"),
    "comfyui_host": "127.0.0.1:8188",
    "comfyui_launcher": "run_nvidia_gpu.bat",
    "llm_provider": "ollama",
    "llm_base_url": "http://localhost:11434",
    "llm_api_key": "",
    "ollama_host": "http://localhost:11434",
    "ollama_model_creative": "gemma4:26b",
    "ollama_model_fast": "gemma4:e4b",
    "kf_width": 2048,
    "kf_height": 1024,
    "video_width": 1024,
    "video_height": 432,
}


def _load_user_settings() -> dict:
    if os.path.exists(_SETTINGS_PATH):
        with open(_SETTINGS_PATH) as f:
            return json.load(f)
    return {}


def save_user_settings(settings: dict):
    merged = _load_user_settings()
    merged.update({k: v for k, v in settings.items() if v is not None})
    with open(_SETTINGS_PATH, "w") as f:
        json.dump(merged, f, indent=2)
    load_runtime_settings()


_user = _load_user_settings()


def _get(key: str) -> str:
    return _user.get(key, _DEFAULTS[key])


def _resolve_path(path_value: str) -> str:
    path = os.path.expanduser(str(path_value).strip())
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    return os.path.abspath(path)


def get_settings_snapshot() -> dict:
    return {
        "comfyui_root": _get("comfyui_root"),
        "project_output_root": _resolve_path(_get("project_output_root")),
        "comfyui_host": _get("comfyui_host"),
        "comfyui_launcher": _get("comfyui_launcher"),
        "llm_provider": _get("llm_provider"),
        "llm_base_url": _get("llm_base_url"),
        "llm_api_key": _get("llm_api_key"),
        "ollama_host": _get("ollama_host"),
        "ollama_model_creative": _get("ollama_model_creative"),
        "ollama_model_fast": _get("ollama_model_fast"),
        "kf_width": int(_get("kf_width")),
        "kf_height": int(_get("kf_height")),
        "video_width": int(_get("video_width")),
        "video_height": int(_get("video_height")),
    }


def _discover_ffmpeg_paths(comfyui_root: str) -> tuple[str, str]:
    ffmpeg_candidates = [
        os.path.join(comfyui_root, "ffmpeg.exe"),
        os.path.join(comfyui_root, "ffmpeg", "ffmpeg.exe"),
    ]
    ffmpeg_path = "ffmpeg"
    for candidate in ffmpeg_candidates:
        if os.path.exists(candidate):
            ffmpeg_path = candidate
            break

    ffprobe_candidates = [
        os.path.join(comfyui_root, "ffprobe.exe"),
        os.path.join(comfyui_root, "ffmpeg", "ffprobe.exe"),
    ]
    for root, _dirs, files in os.walk(comfyui_root):
        for filename in files:
            if filename == "ffprobe.exe":
                ffprobe_candidates.append(os.path.join(root, filename))
        if len(ffprobe_candidates) > 5:
            break

    ffprobe_path = "ffprobe"
    for candidate in ffprobe_candidates:
        if os.path.exists(candidate):
            ffprobe_path = candidate
            break

    return ffmpeg_path, ffprobe_path


def load_runtime_settings() -> dict:
    global _user, _comfyui_root
    global COMFYUI_HOST, COMFYUI_OUTPUT_DIR, COMFYUI_LAUNCHER, COMFYUI_STARTUP_TIMEOUT
    global FFMPEG_PATH, FFPROBE_PATH, OUTPUT_DIR
    global LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, OLLAMA_HOST
    global OLLAMA_MODEL_CREATIVE, OLLAMA_MODEL_FAST, OLLAMA_MODEL
    global KF_WIDTH, KF_HEIGHT, VIDEO_WIDTH, VIDEO_HEIGHT

    _user = _load_user_settings()
    _comfyui_root = _get("comfyui_root")

    OUTPUT_DIR = _resolve_path(_get("project_output_root"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    COMFYUI_HOST = _get("comfyui_host")
    COMFYUI_OUTPUT_DIR = os.path.join(_comfyui_root, "ComfyUI", "output")
    COMFYUI_LAUNCHER = os.path.join(_comfyui_root, _get("comfyui_launcher"))
    COMFYUI_STARTUP_TIMEOUT = 120

    FFMPEG_PATH, FFPROBE_PATH = _discover_ffmpeg_paths(_comfyui_root)

    LLM_PROVIDER = _get("llm_provider")
    LLM_BASE_URL = _get("llm_base_url")
    LLM_API_KEY = _get("llm_api_key")
    OLLAMA_HOST = _get("ollama_host")
    OLLAMA_MODEL_CREATIVE = _get("ollama_model_fast")
    OLLAMA_MODEL_FAST = _get("ollama_model_fast")
    OLLAMA_MODEL = OLLAMA_MODEL_FAST

    KF_WIDTH = int(_get("kf_width"))
    KF_HEIGHT = int(_get("kf_height"))
    VIDEO_WIDTH = int(_get("video_width"))
    VIDEO_HEIGHT = int(_get("video_height"))

    return get_settings_snapshot()


# --- Derived paths from ComfyUI root ---
load_runtime_settings()


def get_output_root() -> str:
    return OUTPUT_DIR


def project_dir(project_name: str) -> str:
    return os.path.join(OUTPUT_DIR, project_name)


def project_state_path(project_name: str) -> str:
    return os.path.join(project_dir(project_name), "state.json")

# --- LTX-AV (video generation) ---
LTX_FPS = 24

# --- Scene duration limits (seconds) ---
SCENE_MIN_SEC = 2
SCENE_MAX_SEC = 30
SCENE_SWEET_SPOT_SEC = 15

# --- Video workflow node IDs (t2v transformer pipeline) ---
PROMPT_NODE_ID = "153:132"
NEG_PROMPT_NODE_ID = "153:123"
FRAMES_NODE_ID = "153:125"
SEED_NODE_ID_PASS1 = "153:151"
SEED_NODE_ID_PASS2 = "153:127"

VIDEO_RES_NODE_ID = "153:124"   # EmptyImage node that sets video resolution

# --- Keyframe image workflow node IDs (Z-Image Turbo) ---
KF_PROMPT_NODE_ID = "57:27"
KF_SEED_NODE_ID = "57:3"
KF_LATENT_NODE_ID = "57:13"

# --- Video resolution ---

# --- Agent behavior ---
TAKES_PER_SCENE = 3
USE_KEYFRAMES = True
KF_CANDIDATES = 4
LAZY_MODE = False                # AI auto-selects best keyframes and takes (no manual review)
SKIP_KF_EVAL = True              # Skip AI evaluation of keyframes (experimental)
SUBTITLE_SAFE_MODE = False       # Avoid literal quoted dialogue in prompts to reduce burned-in captions
EVAL_FRAME_SAMPLE_RATE = 1
EVAL_MAX_FRAMES = 20
EVAL_TOKEN_BUDGET = 1024

# --- Negative prompt ---
NEGATIVE_PROMPT = (
    "background music, soundtrack, musical score, "
    "blurry, low quality, still frame, frozen motion, watermark, overlay, "
    "titles, subtitles, text on screen, deformed hands, extra fingers"
)
