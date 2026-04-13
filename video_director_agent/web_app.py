#!/usr/bin/env python3
"""Minimal web interface for KupkaProd Cinema Pipeline."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import quote

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

sys.path.insert(0, BASE_DIR)

import config
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from agent import load_state, preflight, run, slugify
from assembler import concat_scenes
from config import _get, get_settings_snapshot, load_runtime_settings, save_user_settings

app = FastAPI(title="KupkaProd Web")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@dataclass
class RuntimeOptions:
    lazy: bool
    is_script: bool
    t2v_only: bool
    skip_kf_eval: bool
    subtitle_safe: bool
    takes: int
    scene_min: int
    scene_max: int
    kf_width: int
    kf_height: int
    video_width: int
    video_height: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "lazy": self.lazy,
            "is_script": self.is_script,
            "t2v_only": self.t2v_only,
            "skip_kf_eval": self.skip_kf_eval,
            "subtitle_safe": self.subtitle_safe,
            "takes": self.takes,
            "scene_min": self.scene_min,
            "scene_max": self.scene_max,
            "kf_width": self.kf_width,
            "kf_height": self.kf_height,
            "video_width": self.video_width,
            "video_height": self.video_height,
        }


DEFAULT_PIPELINE_OPTIONS = {
    "takes": 3,
    "scene_min": 2,
    "scene_max": 30,
    "skip_kf_eval": True,
    "subtitle_safe": False,
}


def _output_root() -> str:
    return config.get_output_root()


def _ensure_output_root():
    os.makedirs(_output_root(), exist_ok=True)


_ensure_output_root()


class JobRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._active_project: str | None = None

    def snapshot(self, project_name: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(project_name, {}).copy()
            if "logs" in job:
                job["logs"] = list(job["logs"])
            return job

    def all(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            result = {}
            for name, job in self._jobs.items():
                result[name] = {
                    "running": job.get("running", False),
                    "started_at": job.get("started_at"),
                    "finished_at": job.get("finished_at"),
                    "last_error": job.get("last_error"),
                }
            return result

    def can_start(self, project_name: str) -> tuple[bool, str | None]:
        with self._lock:
            if self._active_project and self._active_project != project_name:
                active = self._jobs.get(self._active_project, {})
                thread = active.get("thread")
                if thread and thread.is_alive():
                    return False, self._active_project
                self._active_project = None
            return True, None

    def begin(self, project_name: str, thread: threading.Thread):
        with self._lock:
            self._active_project = project_name
            job = self._jobs.setdefault(project_name, {})
            job.update({
                "thread": thread,
                "running": True,
                "started_at": time.time(),
                "finished_at": None,
                "last_error": None,
                "logs": [],
            })

    def append_log(self, project_name: str, line: str):
        with self._lock:
            job = self._jobs.setdefault(project_name, {"logs": []})
            logs = job.setdefault("logs", [])
            logs.append(line)
            if len(logs) > 800:
                del logs[: len(logs) - 800]

    def finish(self, project_name: str, error: str | None = None):
        with self._lock:
            job = self._jobs.setdefault(project_name, {})
            job["running"] = False
            job["finished_at"] = time.time()
            job["last_error"] = error
            if self._active_project == project_name:
                self._active_project = None


jobs = JobRegistry()


class ProjectLogHandler(logging.Handler):
    def __init__(self, project_name: str):
        super().__init__()
        self.project_name = project_name

    def emit(self, record):
        try:
            msg = self.format(record)
            jobs.append_log(self.project_name, msg)
        except Exception:
            self.handleError(record)


def _bool_from_form(form, key: str, default: bool = False) -> bool:
    if key not in form:
        return default
    value = str(form.get(key, "")).strip().lower()
    return value not in ("0", "false", "off", "no", "")


def _int_from_form(form, key: str, default: int) -> int:
    value = form.get(key)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _media_url(path: str | None) -> str | None:
    if not path:
        return None
    abs_path = os.path.abspath(path)
    try:
        rel_path = os.path.relpath(abs_path, _output_root())
    except ValueError:
        return None
    if rel_path.startswith(".."):
        return None
    return f"/media/{quote(rel_path.replace(os.sep, '/'))}"


def _state_file(project_name: str) -> str:
    return config.project_state_path(project_name)


def _save_state(project_name: str, state: dict):
    path = _state_file(project_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def _list_projects() -> list[dict[str, Any]]:
    project_names: set[str] = set()
    output_root = _output_root()
    if os.path.exists(output_root):
        for name in os.listdir(output_root):
            if os.path.exists(_state_file(name)):
                project_names.add(name)
    project_names.update(jobs.all().keys())

    summaries = []
    for name in sorted(project_names):
        summaries.append(_project_summary(name))
    summaries.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
    return summaries


def _derive_phase(state: dict | None, job: dict[str, Any]) -> str:
    if job.get("running"):
        return "Running"
    if job.get("last_error"):
        return "Paused with error"
    if not state:
        return "Not started"
    if state.get("completed_at"):
        return "Completed"
    if state.get("generation_completed_at"):
        return "Ready for take review"
    if state.get("storyboard_approved"):
        return "Storyboard approved"
    scenes = state.get("scenes", [])
    if any(scene.get("keyframe_candidates") for scene in scenes):
        return "Storyboard review"
    if scenes:
        return "Scenes planned"
    return "Created"


def _project_summary(project_name: str) -> dict[str, Any]:
    state = load_state(project_name)
    job = jobs.snapshot(project_name)
    scenes = state.get("scenes", []) if state else []
    keyframes_done = sum(1 for scene in scenes if scene.get("keyframe_approved"))
    takes_done = sum(1 for scene in scenes if scene.get("selected_take"))

    updated_at = 0.0
    if state:
        for key in ("completed_at", "generation_completed_at", "storyboard_generated_at", "created_at"):
            value = state.get(key)
            if value:
                try:
                    updated_at = max(updated_at, time.mktime(time.strptime(value[:19], "%Y-%m-%dT%H:%M:%S")))
                except Exception:
                    pass
    updated_at = max(updated_at, job.get("started_at") or 0, job.get("finished_at") or 0)

    return {
        "name": project_name,
        "state": state,
        "job": job,
        "phase": _derive_phase(state, job),
        "total_scenes": len(scenes),
        "keyframes_done": keyframes_done,
        "takes_done": takes_done,
        "final_url": _media_url(state.get("final_path")) if state else None,
        "updated_at": updated_at,
    }


def _project_context(project_name: str) -> dict[str, Any]:
    state = load_state(project_name)
    job = jobs.snapshot(project_name)
    if not state and not job:
        raise HTTPException(status_code=404, detail="Project not found")
    if not state:
        state = {
            "project_name": project_name,
            "brief": "",
            "scenes": [],
        }

    scenes = state.get("scenes", [])
    project_runtime = dict(_current_pipeline_defaults())
    project_runtime.update(state.get("runtime_options", {}))
    storyboard_ready = any(scene.get("keyframe_candidates") for scene in scenes)
    take_review_ready = any(scene.get("takes") for scene in scenes)

    storyboard_scenes = []
    for scene in scenes:
        candidates = []
        for candidate in scene.get("keyframe_candidates", []):
            candidate = dict(candidate)
            candidate["url"] = _media_url(candidate.get("path"))
            candidate["selected"] = scene.get("selected_keyframe") == candidate.get("path")
            candidates.append(candidate)
        storyboard_scenes.append({
            "scene": scene,
            "candidates": candidates,
        })

    take_scenes = []
    for scene in scenes:
        takes = []
        for take in scene.get("takes", []):
            if take.get("status") != "generated":
                continue
            take = dict(take)
            take["url"] = _media_url(take.get("path"))
            take["selected"] = scene.get("selected_take") == take.get("path")
            takes.append(take)
        take_scenes.append({
            "scene": scene,
            "takes": takes,
        })

    approved_keyframes = sum(1 for scene in scenes if scene.get("keyframe_approved"))
    selected_takes = sum(1 for scene in scenes if scene.get("selected_take"))

    return {
        "project_name": project_name,
        "state": state,
        "job": job,
        "phase": _derive_phase(state, job),
        "storyboard_ready": storyboard_ready,
        "take_review_ready": take_review_ready,
        "project_runtime": project_runtime,
        "storyboard_scenes": storyboard_scenes,
        "take_scenes": take_scenes,
        "approved_keyframes": approved_keyframes,
        "selected_takes": selected_takes,
        "final_url": _media_url(state.get("final_path")),
        "logs": "\n".join(job.get("logs", [])),
    }


def _sync_runtime_modules():
    import agent as agent_mod
    import assembler as assembler_mod
    import comfyui_client
    import config
    import director as director_mod
    import evaluator as evaluator_mod
    import keyframe_gen as keyframe_mod
    import llm_client

    snapshot = load_runtime_settings()
    llm_client.clear_model_cache()

    agent_mod.COMFYUI_HOST = config.COMFYUI_HOST
    agent_mod.OLLAMA_MODEL = config.OLLAMA_MODEL
    agent_mod.COMFYUI_LAUNCHER = config.COMFYUI_LAUNCHER
    agent_mod.COMFYUI_STARTUP_TIMEOUT = config.COMFYUI_STARTUP_TIMEOUT
    agent_mod.FFMPEG_PATH = config.FFMPEG_PATH
    agent_mod.OUTPUT_DIR = config.OUTPUT_DIR

    comfyui_client.COMFYUI_HOST = config.COMFYUI_HOST
    comfyui_client.COMFYUI_OUTPUT_DIR = config.COMFYUI_OUTPUT_DIR
    comfyui_client.VIDEO_WIDTH = config.VIDEO_WIDTH
    comfyui_client.VIDEO_HEIGHT = config.VIDEO_HEIGHT
    comfyui_client.NEGATIVE_PROMPT = config.NEGATIVE_PROMPT

    director_mod.OLLAMA_MODEL_CREATIVE = config.OLLAMA_MODEL_CREATIVE
    director_mod.SUBTITLE_SAFE_MODE = config.SUBTITLE_SAFE_MODE
    evaluator_mod.OLLAMA_MODEL_FAST = config.OLLAMA_MODEL_FAST
    keyframe_mod.OLLAMA_MODEL_CREATIVE = config.OLLAMA_MODEL_CREATIVE
    keyframe_mod.OLLAMA_MODEL_FAST = config.OLLAMA_MODEL_FAST
    keyframe_mod.COMFYUI_OUTPUT_DIR = config.COMFYUI_OUTPUT_DIR
    keyframe_mod.KF_WIDTH = config.KF_WIDTH
    keyframe_mod.KF_HEIGHT = config.KF_HEIGHT
    assembler_mod.OLLAMA_MODEL_FAST = config.OLLAMA_MODEL_FAST
    assembler_mod.FFMPEG_PATH = config.FFMPEG_PATH
    assembler_mod.FFPROBE_PATH = config.FFPROBE_PATH
    _ensure_output_root()

    return snapshot


def _current_pipeline_defaults() -> dict[str, Any]:
    snapshot = get_settings_snapshot()
    return {
        "takes": DEFAULT_PIPELINE_OPTIONS["takes"],
        "scene_min": DEFAULT_PIPELINE_OPTIONS["scene_min"],
        "scene_max": DEFAULT_PIPELINE_OPTIONS["scene_max"],
        "skip_kf_eval": DEFAULT_PIPELINE_OPTIONS["skip_kf_eval"],
        "subtitle_safe": DEFAULT_PIPELINE_OPTIONS["subtitle_safe"],
        "kf_width": snapshot["kf_width"],
        "kf_height": snapshot["kf_height"],
        "video_width": snapshot["video_width"],
        "video_height": snapshot["video_height"],
    }


def _apply_runtime_overrides(options: RuntimeOptions):
    import agent as agent_mod
    import comfyui_client
    import config
    import director as director_mod
    import keyframe_gen as keyframe_mod

    config.KF_WIDTH = options.kf_width
    config.KF_HEIGHT = options.kf_height
    config.VIDEO_WIDTH = options.video_width
    config.VIDEO_HEIGHT = options.video_height
    keyframe_mod.KF_WIDTH = options.kf_width
    keyframe_mod.KF_HEIGHT = options.kf_height
    comfyui_client.VIDEO_WIDTH = options.video_width
    comfyui_client.VIDEO_HEIGHT = options.video_height

    config.TAKES_PER_SCENE = options.takes
    agent_mod.TAKES_PER_SCENE = options.takes
    config.SCENE_MIN_SEC = options.scene_min
    config.SCENE_MAX_SEC = options.scene_max
    director_mod.SCENE_MIN_SEC = options.scene_min
    director_mod.SCENE_MAX_SEC = options.scene_max

    config.USE_KEYFRAMES = not options.t2v_only
    agent_mod.USE_KEYFRAMES = not options.t2v_only
    config.SKIP_KF_EVAL = options.skip_kf_eval
    config.SUBTITLE_SAFE_MODE = options.subtitle_safe
    keyframe_mod.SKIP_KF_EVAL = options.skip_kf_eval
    director_mod.SUBTITLE_SAFE_MODE = options.subtitle_safe
    comfyui_client.COMFYUI_HOST = config.COMFYUI_HOST
    comfyui_client.COMFYUI_OUTPUT_DIR = config.COMFYUI_OUTPUT_DIR
    keyframe_mod.COMFYUI_OUTPUT_DIR = config.COMFYUI_OUTPUT_DIR


def _restart_ollama_if_needed(log: logging.Logger):
    import subprocess
    import urllib.request

    from llm_client import is_ollama_provider, provider_label

    if not is_ollama_provider():
        log.info("Skipping Ollama restart because LLM provider is %s.", provider_label())
        return

    log.info("Restarting Ollama...")
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/f", "/im", "ollama.exe"], capture_output=True, timeout=10)
            subprocess.run(["taskkill", "/f", "/im", "ollama_llama_server.exe"], capture_output=True, timeout=10)
        else:
            subprocess.run(["pkill", "-f", "ollama"], capture_output=True, timeout=10)
        time.sleep(2)
    except Exception as exc:
        log.warning("Could not stop Ollama cleanly: %s", exc)

    try:
        kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        subprocess.Popen(["ollama", "serve"], **kwargs)
    except FileNotFoundError:
        log.warning("ollama not found on PATH, skipping restart")
        return

    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as response:
                if response.status == 200:
                    log.info("Ollama restarted and ready.")
                    return
        except Exception:
            time.sleep(1)
    log.warning("Ollama did not respond within 30s after restart")


def _run_project_background(project_name: str, brief: str, options: RuntimeOptions):
    handler = ProjectLogHandler(project_name)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    error_message = None
    try:
        from comfyui_client import ComfyUIClient

        _apply_runtime_overrides(options)
        log = logging.getLogger("agent")
        _restart_ollama_if_needed(log)
        client = ComfyUIClient()
        preflight(client, log)
        run(brief, project_name, log, is_script=options.is_script, lazy=options.lazy)
    except SystemExit:
        error_message = "Preflight failed"
        jobs.append_log(project_name, error_message)
    except Exception as exc:
        error_message = str(exc)
        jobs.append_log(project_name, f"ERROR: {exc}")
    finally:
        root_logger.removeHandler(handler)
        jobs.finish(project_name, error_message)


def _start_project(project_name: str, brief: str, options: RuntimeOptions):
    allowed, active_project = jobs.can_start(project_name)
    if not allowed:
        raise RuntimeError(f"Project '{active_project}' is still running. Only one active production job is supported in this first web version.")

    existing = jobs.snapshot(project_name)
    thread = existing.get("thread")
    if thread and thread.is_alive():
        return

    state = load_state(project_name) or {
        "project_name": project_name,
        "brief": brief,
        "created_at": datetime.now().isoformat(),
        "total_scenes": 0,
        "scenes": [],
    }
    state["brief"] = brief
    state.setdefault("created_at", datetime.now().isoformat())
    state.setdefault("total_scenes", len(state.get("scenes", [])))
    state.setdefault("scenes", [])
    state.setdefault("is_script", options.is_script)
    state["runtime_options"] = options.as_dict()
    _save_state(project_name, state)

    worker = threading.Thread(
        target=_run_project_background,
        args=(project_name, brief, options),
        daemon=True,
        name=f"kupka-web-{project_name}",
    )
    jobs.begin(project_name, worker)
    worker.start()


def _build_runtime_options(form) -> RuntimeOptions:
    return RuntimeOptions(
        lazy=_bool_from_form(form, "lazy", False),
        is_script=_bool_from_form(form, "is_script", False),
        t2v_only=_bool_from_form(form, "t2v_only", False),
        skip_kf_eval=_bool_from_form(form, "skip_kf_eval", True),
        subtitle_safe=_bool_from_form(form, "subtitle_safe", False),
        takes=_int_from_form(form, "takes", 3),
        scene_min=_int_from_form(form, "scene_min", 2),
        scene_max=_int_from_form(form, "scene_max", 30),
        kf_width=_int_from_form(form, "kf_width", 2048),
        kf_height=_int_from_form(form, "kf_height", 1024),
        video_width=_int_from_form(form, "video_width", 1024),
        video_height=_int_from_form(form, "video_height", 432),
    )


def _save_settings_from_form(form):
    settings = {
        "comfyui_root": form.get("comfyui_root", _get("comfyui_root")),
        "project_output_root": form.get("project_output_root", _get("project_output_root")),
        "comfyui_launcher": form.get("comfyui_launcher", _get("comfyui_launcher")),
        "llm_provider": form.get("llm_provider", _get("llm_provider")),
        "llm_base_url": form.get("llm_base_url", _get("llm_base_url")),
        "ollama_host": form.get("llm_base_url", _get("llm_base_url")),
        "ollama_model_creative": form.get("llm_model_creative", _get("ollama_model_creative")),
        "ollama_model_fast": form.get("llm_model_fast", _get("ollama_model_fast")),
        "kf_width": _int_from_form(form, "kf_width", int(_get("kf_width"))),
        "kf_height": _int_from_form(form, "kf_height", int(_get("kf_height"))),
        "video_width": _int_from_form(form, "video_width", int(_get("video_width"))),
        "video_height": _int_from_form(form, "video_height", int(_get("video_height"))),
    }
    save_user_settings(settings)
    return _sync_runtime_modules()


def _base_template_context(request: Request) -> dict[str, Any]:
    settings = get_settings_snapshot()
    return {
        "request": request,
        "settings": {
            "comfyui_root": settings["comfyui_root"],
            "project_output_root": settings["project_output_root"],
            "comfyui_launcher": settings["comfyui_launcher"],
            "llm_provider": settings["llm_provider"],
            "llm_base_url": settings["llm_base_url"],
            "llm_model_creative": settings["ollama_model_creative"],
            "llm_model_fast": settings["ollama_model_fast"],
        },
        "runtime_defaults": _current_pipeline_defaults(),
    }


@app.get("/media/{media_path:path}")
def media_file(media_path: str):
    base = _output_root()
    abs_path = os.path.abspath(os.path.join(base, media_path))
    try:
        if os.path.commonpath([base, abs_path]) != base:
            raise HTTPException(status_code=404, detail="Media not found")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Media not found") from exc
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(abs_path)


@app.get("/", response_class=HTMLResponse)
def index(request: Request, message: str | None = None, error: str | None = None):
    context = _base_template_context(request)
    context.update({
        "projects": _list_projects(),
        "message": message,
        "error": error,
    })
    return templates.TemplateResponse("index.html", context)


@app.post("/settings/save")
async def save_settings(request: Request):
    form = await request.form()
    _save_settings_from_form(form)
    return RedirectResponse(url="/?message=Settings+saved", status_code=303)


@app.post("/projects/start")
async def start_project(request: Request):
    form = await request.form()
    _save_settings_from_form(form)

    brief = str(form.get("brief", "")).strip()
    project_name = str(form.get("project_name", "")).strip()
    project_name = project_name or slugify(brief)

    if not project_name:
        return RedirectResponse(url="/?error=Please+provide+a+project+name+or+brief.", status_code=303)

    options = _build_runtime_options(form)

    if not brief:
        existing_state = load_state(project_name)
        if not existing_state:
            return RedirectResponse(url=f"/?error=Project+{quote(project_name)}+has+no+saved+state+to+resume.", status_code=303)
        brief = existing_state["brief"]

    try:
        _start_project(project_name, brief, options)
    except RuntimeError as exc:
        return RedirectResponse(url=f"/?error={quote(str(exc))}", status_code=303)

    return RedirectResponse(url=f"/projects/{quote(project_name)}?message=Production+started", status_code=303)


@app.get("/projects/{project_name}", response_class=HTMLResponse)
def project_detail(request: Request, project_name: str, message: str | None = None):
    context = _base_template_context(request)
    context.update(_project_context(project_name))
    context["message"] = message
    return templates.TemplateResponse("project.html", context)


@app.get("/projects/{project_name}/api")
def project_api(project_name: str):
    context = _project_context(project_name)
    state = context["state"]
    scenes = state.get("scenes", [])
    approved_keyframes = sum(1 for scene in scenes if scene.get("keyframe_approved"))
    selected_takes = sum(1 for scene in scenes if scene.get("selected_take"))
    return JSONResponse({
        "phase": context["phase"],
        "running": context["job"].get("running", False),
        "error": context["job"].get("last_error"),
        "logs": context["logs"],
        "storyboard_approved": bool(state.get("storyboard_approved")),
        "generation_completed": bool(state.get("generation_completed_at")),
        "completed": bool(state.get("completed_at")),
        "total_scenes": len(scenes),
        "approved_keyframes": approved_keyframes,
        "selected_takes": selected_takes,
        "final_url": context["final_url"],
    })


def _generate_redo_takes(project_name: str, scene_number: int, count: int, clear_old: bool):
    from comfyui_client import (
        ComfyUIClient, build_i2v_workflow, build_workflow, calc_frames,
        load_i2v_template, load_workflow_template,
    )
    from config import COMFYUI_LAUNCHER, COMFYUI_STARTUP_TIMEOUT, LTX_FPS

    handler = ProjectLogHandler(project_name)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    error_message = None
    client = None
    try:
        _sync_runtime_modules()
        log = logging.getLogger("web.redo")
        state = load_state(project_name)
        if not state:
            raise RuntimeError("Project state not found")

        scene = next((item for item in state.get("scenes", []) if int(item.get("scene_number", 0)) == int(scene_number)), None)
        if scene is None:
            raise RuntimeError(f"Scene {scene_number} not found")

        if clear_old:
            scene["takes"] = []
        scene.pop("selected_take", None)
        scene.pop("status", None)
        scene["takes_done"] = False
        scene["redo_takes"] = count
        _save_state(project_name, state)

        jobs.append_log(project_name, f"Generating {count} new take(s) for scene {scene_number}...")
        client = ComfyUIClient()

        if not client.check_alive():
            import subprocess as _sp

            jobs.append_log(project_name, "ComfyUI not running. Launching...")
            kwargs = {"cwd": os.path.dirname(COMFYUI_LAUNCHER)}
            if os.name == "nt":
                kwargs["creationflags"] = _sp.CREATE_NEW_PROCESS_GROUP
            _sp.Popen(COMFYUI_LAUNCHER, **kwargs)
            deadline = time.time() + COMFYUI_STARTUP_TIMEOUT
            while time.time() < deadline:
                if client.check_alive():
                    break
                time.sleep(3)
            else:
                raise RuntimeError("ComfyUI did not start in time")

        client.connect()
        template = load_workflow_template()
        i2v_template = load_i2v_template()
        project_dir = config.project_dir(project_name)
        scenes_dir = os.path.join(project_dir, "scenes")
        os.makedirs(scenes_dir, exist_ok=True)

        frames = calc_frames(scene["duration_seconds"], LTX_FPS)
        keyframe_path = scene.get("selected_keyframe")
        use_i2v = i2v_template and keyframe_path and os.path.exists(keyframe_path)
        uploaded_kf_name = None
        if use_i2v:
            try:
                uploaded_kf_name = client.upload_image(keyframe_path)
                log.info("Using I2V mode with keyframe %s", os.path.basename(keyframe_path))
            except Exception as exc:
                log.warning("Failed to upload keyframe: %s — falling back to T2V", exc)
                use_i2v = False

        existing_count = len(scene.get("takes", []))
        new_takes = []
        for index in range(1, count + 1):
            take_num = existing_count + index
            seed = random.randint(0, 2**32 - 1)
            jobs.append_log(project_name, f"Scene {scene_number}: generating take {index}/{count} ({'i2v' if use_i2v else 't2v'})")

            if use_i2v:
                workflow = build_i2v_workflow(i2v_template, scene["ltx_prompt"], frames, seed, uploaded_kf_name)
            else:
                workflow = build_workflow(template, scene["ltx_prompt"], frames, seed)

            try:
                prompt_id = client.queue_prompt(workflow)
                history = client.wait_for_completion(prompt_id, timeout=900)
                raw_output = client.get_output_path(history)
            except Exception as exc:
                log.error("Redo take %d failed: %s", take_num, exc)
                new_takes.append({"take": take_num, "status": "failed", "error": str(exc)})
                continue

            take_filename = f"scene_{scene_number:03d}_take_{take_num}.mp4"
            take_path = os.path.join(scenes_dir, take_filename)
            shutil.copy2(raw_output, take_path)
            new_takes.append({
                "take": take_num,
                "status": "generated",
                "path": take_path,
                "seed": seed,
            })

        scene.setdefault("takes", []).extend(new_takes)
        scene["takes_done"] = True
        scene.pop("redo_takes", None)
        _save_state(project_name, state)
        generated = sum(1 for take in new_takes if take.get("status") == "generated")
        jobs.append_log(project_name, f"Scene {scene_number}: done, {generated} new take(s) generated.")
    except Exception as exc:
        error_message = str(exc)
        jobs.append_log(project_name, f"ERROR: {exc}")
    finally:
        if client:
            try:
                client.disconnect()
            except Exception:
                pass
        root_logger.removeHandler(handler)
        jobs.finish(project_name, error_message)


@app.post("/projects/{project_name}/storyboard/select")
async def storyboard_select(project_name: str, request: Request):
    form = await request.form()
    selected_path = str(form.get("candidate_path", ""))
    state = load_state(project_name)
    if not state:
        raise HTTPException(status_code=404, detail="Project not found")

    for scene in state.get("scenes", []):
        if str(scene.get("scene_number")) == str(form.get("scene_number")):
            scene["selected_keyframe"] = selected_path
            scene["keyframe_approved"] = True
            break
    _save_state(project_name, state)
    return RedirectResponse(url=f"/projects/{quote(project_name)}?message=Keyframe+selected", status_code=303)


@app.post("/projects/{project_name}/storyboard/reject")
async def storyboard_reject(project_name: str, request: Request):
    form = await request.form()
    state = load_state(project_name)
    if not state:
        raise HTTPException(status_code=404, detail="Project not found")

    scene_number = str(form.get("scene_number"))
    notes = str(form.get("rejection_notes", "")).strip()
    for scene in state.get("scenes", []):
        if str(scene.get("scene_number")) == scene_number:
            scene["rejection_notes"] = notes or "Rejected from web UI"
            scene["keyframe_approved"] = False
            scene.pop("selected_keyframe", None)
            scene.pop("keyframe_candidates", None)
            break
    _save_state(project_name, state)
    return RedirectResponse(url=f"/projects/{quote(project_name)}?message=Scene+marked+for+keyframe+regen", status_code=303)


@app.post("/projects/{project_name}/storyboard/proceed")
def storyboard_proceed(project_name: str):
    state = load_state(project_name)
    if not state:
        raise HTTPException(status_code=404, detail="Project not found")
    state["storyboard_approved"] = True
    _save_state(project_name, state)
    return RedirectResponse(url=f"/projects/{quote(project_name)}?message=Storyboard+approved.+Resume+production+to+generate+takes.", status_code=303)


@app.post("/projects/{project_name}/takes/select")
async def take_select(project_name: str, request: Request):
    form = await request.form()
    take_path = str(form.get("take_path", ""))
    scene_number = str(form.get("scene_number"))
    state = load_state(project_name)
    if not state:
        raise HTTPException(status_code=404, detail="Project not found")

    for scene in state.get("scenes", []):
        if str(scene.get("scene_number")) == scene_number:
            scene["selected_take"] = take_path
            scene["status"] = "approved"
            break
    _save_state(project_name, state)
    return RedirectResponse(url=f"/projects/{quote(project_name)}?message=Take+selected", status_code=303)


@app.post("/projects/{project_name}/takes/regenerate")
async def regenerate_takes(project_name: str, request: Request):
    form = await request.form()
    scene_number = _int_from_form(form, "scene_number", 0)
    count = _int_from_form(form, "redo_count", 3)
    clear_old = _bool_from_form(form, "clear_old", True)

    if scene_number <= 0:
        return RedirectResponse(url=f"/projects/{quote(project_name)}?message=Invalid+scene+for+take+regen", status_code=303)

    allowed, active_project = jobs.can_start(project_name)
    if not allowed:
        return RedirectResponse(
            url=f"/projects/{quote(project_name)}?message=Project+{quote(active_project)}+is+already+running",
            status_code=303,
        )

    worker = threading.Thread(
        target=_generate_redo_takes,
        args=(project_name, scene_number, count, clear_old),
        daemon=True,
        name=f"kupka-redo-{project_name}-{scene_number}",
    )
    jobs.begin(project_name, worker)
    worker.start()
    return RedirectResponse(url=f"/projects/{quote(project_name)}?message=Take+regeneration+started", status_code=303)


@app.post("/projects/{project_name}/assemble")
def assemble_project(project_name: str):
    state = load_state(project_name)
    if not state:
        raise HTTPException(status_code=404, detail="Project not found")

    paths = []
    for scene in state.get("scenes", []):
        path = scene.get("selected_take")
        if not path or not os.path.exists(path):
            return RedirectResponse(
                url=f"/projects/{quote(project_name)}?message=Every+scene+needs+a+selected+take+before+assembly.",
                status_code=303,
            )
        paths.append(path)

    final_dir = config.project_dir(project_name)
    final_path = os.path.join(final_dir, "final.mp4")
    concat_scenes(paths, final_path)
    state["final_path"] = final_path
    state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _save_state(project_name, state)
    return RedirectResponse(url=f"/projects/{quote(project_name)}?message=Final+film+assembled", status_code=303)
