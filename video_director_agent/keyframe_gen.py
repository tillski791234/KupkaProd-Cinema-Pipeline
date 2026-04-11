# keyframe_gen.py -- Keyframe image generation + AI evaluation via Z-Image Turbo

import json
import logging
import os
import random
import shutil
import base64
import copy

from config import (
    COMFYUI_OUTPUT_DIR, OLLAMA_MODEL_FAST,
    KF_PROMPT_NODE_ID, KF_SEED_NODE_ID, KF_LATENT_NODE_ID,
    KF_CANDIDATES, KF_WIDTH, KF_HEIGHT, SKIP_KF_EVAL,
)
from llm_client import chat as llm_chat

log = logging.getLogger(__name__)


def _crop_to_video_ar(image_path: str, target_w: int = 1024, target_h: int = 432):
    """Crop a square image to the video's widescreen aspect ratio (center crop)."""
    from PIL import Image as PILImage
    img = PILImage.open(image_path)
    w, h = img.size

    # Calculate crop box for target aspect ratio
    target_ar = target_w / target_h  # e.g. 2.37:1
    current_ar = w / h

    if current_ar < target_ar:
        # Image is too tall, crop top and bottom
        new_h = int(w / target_ar)
        top = (h - new_h) // 2
        box = (0, top, w, top + new_h)
    else:
        # Image is too wide, crop sides
        new_w = int(h * target_ar)
        left = (w - new_w) // 2
        box = (left, 0, left + new_w, h)

    cropped = img.crop(box)
    cropped.save(image_path)


def load_keyframe_template(path: str = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "keyframe_template.json")
    with open(path) as f:
        return json.load(f)


def _detect_keyframe_nodes(wf: dict) -> dict:
    """Auto-detect keyframe workflow node IDs from the template."""
    detected = {
        "prompt": KF_PROMPT_NODE_ID,
        "seed": KF_SEED_NODE_ID,
        "latent": KF_LATENT_NODE_ID,
    }

    all_present = all(k in wf for k in [KF_PROMPT_NODE_ID, KF_SEED_NODE_ID, KF_LATENT_NODE_ID])
    if all_present:
        return detected

    log.info("Default keyframe node IDs not found — auto-detecting...")

    # Find prompt node (CLIPTextEncode)
    clip_nodes = [nid for nid, n in wf.items()
                  if isinstance(n, dict) and n.get("class_type") == "CLIPTextEncode"]
    if clip_nodes:
        detected["prompt"] = clip_nodes[0]

    # Find seed node (KSampler or KSamplerAdvanced)
    for cls in ["KSampler", "KSamplerAdvanced", "RandomNoise"]:
        seed_nodes = [nid for nid, n in wf.items()
                      if isinstance(n, dict) and n.get("class_type") == cls]
        if seed_nodes:
            detected["seed"] = seed_nodes[0]
            break

    # Find latent node (EmptySD3LatentImage, EmptyLatentImage, etc.)
    for cls in ["EmptySD3LatentImage", "EmptyLatentImage", "EmptyImage"]:
        latent_nodes = [nid for nid, n in wf.items()
                        if isinstance(n, dict) and n.get("class_type") == cls]
        if latent_nodes:
            detected["latent"] = latent_nodes[0]
            break

    log.info("Auto-detected keyframe node IDs: %s", detected)
    return detected


def build_keyframe_workflow(template: dict, prompt_text: str, seed: int,
                            width: int = 1024, height: int = 432) -> dict:
    """Build a keyframe image workflow with the given prompt and seed.
    Auto-detects node IDs if the configured defaults don't match the template."""
    wf = copy.deepcopy(template)
    nodes = _detect_keyframe_nodes(wf)
    wf[nodes["prompt"]]["inputs"]["text"] = prompt_text
    wf[nodes["seed"]]["inputs"]["seed"] = seed
    wf[nodes["latent"]]["inputs"]["width"] = width
    wf[nodes["latent"]]["inputs"]["height"] = height
    return wf


def get_image_output_path(history: dict, output_dir: str = COMFYUI_OUTPUT_DIR) -> str:
    """Extract image path from ComfyUI history."""
    outputs = history.get("outputs", {})
    for node_id, node_output in outputs.items():
        if "images" in node_output:
            for item in node_output["images"]:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename", "")
                if filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
                    subfolder = item.get("subfolder", "")
                    return os.path.join(output_dir, subfolder, filename)
    raise ValueError("No image output found in history")


def image_to_base64(image_path: str) -> str:
    """Read an image file and return base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def evaluate_keyframe(image_path: str, scene: dict, characters: dict) -> dict:
    """Thoroughly evaluate a keyframe image against the scene + character descriptions.

    Returns dict with scores and verdict.
    """
    img_b64 = image_to_base64(image_path)

    # Build character checklist
    chars_in_scene = scene.get("characters_in_scene", [])
    char_checklist = ""
    if chars_in_scene and characters:
        char_checklist = "\n\nCHARACTER CHECKLIST -- verify EACH of these against the image:\n"
        for char_id in chars_in_scene:
            desc = characters.get(char_id, "")
            if desc:
                char_checklist += f"\n{char_id}:\n{desc}\n"
                char_checklist += "Check: face shape? skin tone? hair color/style? clothing match? age range? build?\n"

    eval_prompt = f"""You are a STRICT storyboard quality control reviewer. Your job is to REJECT images that don't match the character descriptions or scene requirements. Do NOT approve mediocre images.

SCENE REQUIREMENTS:
- Description: {scene['description']}
- Shot type: {scene.get('shot_type', 'not specified')}
- Setting: {scene.get('setting_description', 'not specified')}
- Lighting: {scene.get('lighting_description', 'not specified')}
- Mood: {scene.get('mood', 'not specified')}
{char_checklist}

EVALUATE EACH CATEGORY -- score as "good", "fair", or "poor":

1. CHARACTER_ACCURACY: Does each character match their description?
   - Compare face, skin tone, hair, age, build against the description POINT BY POINT
   - Is the clothing exactly right? (color, style, accessories)
   - FAIL if: wrong face features, wrong hair color, wrong clothing, wrong age range
   - FAIL if: the person is clearly not who they're supposed to be

2. SETTING_ACCURACY: Does the environment match?
   - Background, room type, props, furniture
   - FAIL if: completely wrong location or setting

3. COMPOSITION: Is the shot type correct?
   - Camera angle, framing, subject placement
   - FAIL if: requested close-up but got wide shot, etc.

4. LIGHTING_MOOD: Does the lighting/atmosphere match?
   - Color temperature, shadow direction, mood
   - Minor variations are OK

5. IMAGE_QUALITY: Technical quality
   - No distorted faces, extra limbs, text artifacts, blurriness
   - FAIL if: deformed features, extra fingers, melted faces

VERDICT RULES:
- CHARACTER_ACCURACY "poor" = automatic FAIL (this is the most important)
- ANY other category "poor" = FAIL
- TWO or more "fair" = FAIL

Think carefully about each point before scoring.

Respond with valid JSON:
{{"character_accuracy": "good|fair|poor", "setting_accuracy": "good|fair|poor",
"composition": "good|fair|poor", "lighting_mood": "good|fair|poor",
"image_quality": "good|fair|poor", "verdict": "PASS|FAIL",
"fail_reason": "specific description of what's wrong, or null",
"character_notes": "what specifically matches or doesn't match the character description"}}"""

    response = llm_chat(
        model=OLLAMA_MODEL_FAST,
        messages=[{"role": "user", "content": eval_prompt, "images": [img_b64]}],
        options={
            "num_predict": 1024,
            "temperature": 0.2,  # Very analytical
        },
    )

    raw = response["message"]["content"].strip()
    try:
        # Find JSON in response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(raw[start:end])
        else:
            raise ValueError("No JSON found")

        result["verdict"] = result.get("verdict", "FAIL").upper()
        if result["verdict"] not in ("PASS", "FAIL"):
            result["verdict"] = "FAIL"

        # Enforce rules programmatically
        result = _enforce_keyframe_rules(result)
        return result

    except (json.JSONDecodeError, ValueError):
        log.warning("Keyframe eval parse failed: %s", raw[:200])
        return {
            "character_accuracy": "unknown", "setting_accuracy": "unknown",
            "composition": "unknown", "lighting_mood": "unknown",
            "image_quality": "unknown", "verdict": "FAIL",
            "fail_reason": "Evaluation failed to parse",
            "character_notes": None,
        }


def _enforce_keyframe_rules(result: dict) -> dict:
    """Programmatically enforce FAIL conditions."""
    categories = ["character_accuracy", "setting_accuracy", "composition",
                   "lighting_mood", "image_quality"]

    # Character accuracy poor = instant fail
    if result.get("character_accuracy") == "poor":
        result["verdict"] = "FAIL"
        if not result.get("fail_reason"):
            result["fail_reason"] = "Character doesn't match description"

    poor_count = sum(1 for c in categories if result.get(c) == "poor")
    fair_count = sum(1 for c in categories if result.get(c) == "fair")

    if poor_count > 0 or fair_count >= 2:
        if result.get("verdict") == "PASS":
            reasons = []
            if poor_count:
                poor_cats = [c for c in categories if result.get(c) == "poor"]
                reasons.append(f"poor: {', '.join(poor_cats)}")
            if fair_count >= 2:
                fair_cats = [c for c in categories if result.get(c) == "fair"]
                reasons.append(f"multiple fair: {', '.join(fair_cats)}")
            result["verdict"] = "FAIL"
            result["fail_reason"] = f"Auto-failed: {'; '.join(reasons)}"

    return result


def _rewrite_keyframe_prompt(scene: dict, fail_reasons: list[str], original_prompt: str) -> str:
    """Ask the LLM to rewrite the image prompt based on what kept failing."""
    reasons_text = "\n".join(f"  - Attempt {i+1}: {r}" for i, r in enumerate(fail_reasons))

    rewrite_request = f"""The following image prompt was used to generate keyframe images for a scene, but it FAILED evaluation {len(fail_reasons)} times in a row.

ORIGINAL PROMPT:
{original_prompt}

FAILURE REASONS:
{reasons_text}

SCENE DESCRIPTION: {scene['description']}

Rewrite the prompt to fix these issues. Focus on:
- If characters didn't match: emphasize their physical description more prominently
- If setting was wrong: be more explicit about the environment
- If composition was bad: specify the camera angle and framing more clearly
- Simplify overly complex descriptions that the image model can't handle
- Keep it to one clear moment — don't describe sequential actions
- This is a SILENT storyboard image, not a video/audio prompt
- Do NOT include dialogue quotes, captions, subtitles, speech bubbles, lower thirds, or any readable text

Respond with ONLY the rewritten prompt text, nothing else."""

    response = llm_chat(
        model=OLLAMA_MODEL_FAST,
        messages=[{"role": "user", "content": rewrite_request}],
        options={"num_predict": 2048, "temperature": 0.5},
    )
    return response["message"]["content"].strip()


def _keyframe_prompt_suffix() -> str:
    return (
        "\n\nSTORYBOARD FRAME RULES:\n"
        "- This is a single silent storyboard frame for an image model.\n"
        "- Show only the visual moment, composition, characters, setting, lighting, and action.\n"
        "- Do NOT render subtitles, captions, speech bubbles, lower thirds, signs, readable labels, or any other readable text.\n"
        "- Do NOT show written dialogue anywhere in the image.\n"
    )


def _build_keyframe_prompt(scene: dict, brief: str = "") -> str:
    """Create a visual-only prompt for keyframe generation."""
    from director import write_prompt

    image_scene = copy.deepcopy(scene)
    image_scene["dialogue"] = ""
    image_scene["audio_description"] = ""
    base_prompt = write_prompt(image_scene, brief=brief).strip()
    return base_prompt + _keyframe_prompt_suffix()


def _needs_keyframe_prompt_refresh(scene: dict) -> bool:
    prompt = scene.get("keyframe_prompt", "")
    if not prompt:
        return True
    return "STORYBOARD FRAME RULES:" not in prompt


def _run_keyframe_round(client, template: dict, scene: dict, characters: dict,
                        keyframe_dir: str, prompt: str, candidates: list,
                        max_attempts: int, scene_num: int) -> bool:
    """Run a round of keyframe generation attempts. Returns True if one passed."""
    seed = random.randint(0, 2**32 - 1)
    start_idx = len(candidates)

    for i in range(max_attempts):
        candidate_num = start_idx + i + 1
        log.info("  Keyframe %d for scene %d (seed %d)...",
                 candidate_num, scene_num, seed)

        workflow = build_keyframe_workflow(
            template, prompt, seed,
            width=KF_WIDTH, height=KF_HEIGHT,
        )

        try:
            prompt_id = client.queue_prompt(workflow)
            history = client.wait_for_completion(prompt_id, timeout=120)
            raw_path = get_image_output_path(history)
        except Exception as e:
            log.error("  Keyframe gen failed: %s", e)
            candidates.append({"candidate": candidate_num, "status": "failed", "error": str(e)})
            seed = random.randint(0, 2**32 - 1)
            continue

        kf_filename = f"scene_{scene_num:03d}_kf_{candidate_num}.png"
        kf_path = os.path.join(keyframe_dir, kf_filename)
        shutil.copy2(raw_path, kf_path)

        if SKIP_KF_EVAL:
            log.info("  Keyframe %d generated (evaluation skipped)", candidate_num)
            eval_result = {"verdict": "PASS", "notes": "Evaluation skipped"}
        else:
            log.info("  Evaluating keyframe %d...", candidate_num)
            eval_result = evaluate_keyframe(kf_path, scene, characters)
            log.info("  Keyframe %d: %s -- %s",
                     candidate_num, eval_result["verdict"],
                     eval_result.get("fail_reason") or eval_result.get("character_notes", "OK"))

        candidates.append({
            "candidate": candidate_num,
            "status": "generated",
            "path": kf_path,
            "seed": seed,
            "eval": eval_result,
        })

        if eval_result["verdict"] == "PASS":
            log.info("  Keyframe PASSED -- moving on from scene %d", scene_num)
            return True

        seed = random.randint(0, 2**32 - 1)

    return False


def generate_keyframes(client, scene: dict, characters: dict,
                       project_dir: str, brief: str = "") -> list[dict]:
    """Generate keyframe candidates for a scene with prompt rewriting on failure.

    Round 1: Try KF_CANDIDATES attempts with the original prompt.
    If all fail, rewrite the prompt based on failure reasons and try again.
    """
    template = load_keyframe_template()
    scene_num = scene["scene_number"]

    # Write an image-specific prompt that strips dialogue to avoid burned-in captions.
    if _needs_keyframe_prompt_refresh(scene):
        scene["keyframe_prompt"] = _build_keyframe_prompt(scene, brief=brief)

    prompt = scene["keyframe_prompt"]
    log.info("  Image prompt (%d words):", len(prompt.split()))
    for line in prompt.split("\n"):
        log.info("    | %s", line)

    keyframe_dir = os.path.join(project_dir, "keyframes")
    os.makedirs(keyframe_dir, exist_ok=True)

    candidates = []

    # Round 1: original prompt
    log.info("  Round 1: trying %d candidates with original prompt", KF_CANDIDATES)
    if _run_keyframe_round(client, template, scene, characters,
                           keyframe_dir, prompt, candidates,
                           KF_CANDIDATES, scene_num):
        return candidates

    # All failed — collect failure reasons and rewrite the prompt
    fail_reasons = []
    for c in candidates:
        reason = c.get("eval", {}).get("fail_reason") or c.get("error", "unknown")
        if reason:
            fail_reasons.append(reason)

    log.info("  All %d candidates failed. Rewriting prompt based on failures...", KF_CANDIDATES)
    new_prompt = _rewrite_keyframe_prompt(scene, fail_reasons, prompt)
    scene["keyframe_prompt"] = new_prompt

    log.info("  Rewritten prompt (%d words):", len(new_prompt.split()))
    for line in new_prompt.split("\n"):
        log.info("    | %s", line)

    # Round 2: rewritten prompt
    log.info("  Round 2: trying %d candidates with rewritten prompt", KF_CANDIDATES)
    _run_keyframe_round(client, template, scene, characters,
                        keyframe_dir, new_prompt, candidates,
                        KF_CANDIDATES, scene_num)

    return candidates
