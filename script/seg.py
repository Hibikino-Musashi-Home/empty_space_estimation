import os
import re
import json
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageColor
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types
import dataclasses

# Define the target directory for saving segmentation results
TARGET_DIR = r'../image/'

@dataclasses.dataclass(frozen=True)
class SegmentationMask:
    y0: int
    x0: int
    y1: int
    x1: int
    mask: np.ndarray
    label: str


def extract_json(json_output: str) -> str:
    # Remove markdown fencing to extract raw JSON
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output


def get_next_segmentation_image_index() -> int:
    # Compute next index based on existing files
    os.makedirs(TARGET_DIR, exist_ok=True)
    pattern = re.compile(r'^segmentation_result_(\d+)\.jpg$')
    max_idx = 0
    for fname in os.listdir(TARGET_DIR):
        m = pattern.match(fname)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1


def parse_segmentation_masks(predicted_str: str, *, img_height: int, img_width: int) -> list[SegmentationMask]:
    items = json.loads(extract_json(predicted_str))
    masks: list[SegmentationMask] = []
    for item in items:
        box = item["box_2d"]
        y0 = int(box[0] / 1000 * img_height)
        x0 = int(box[1] / 1000 * img_width)
        y1 = int(box[2] / 1000 * img_height)
        x1 = int(box[3] / 1000 * img_width)
        png_str = item.get("mask", "")
        if not png_str.startswith("data:image/png;base64,"):
            continue
        png_data = base64.b64decode(png_str.removeprefix("data:image/png;base64,"))
        mask_img = Image.open(BytesIO(png_data))
        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR
        mask_img = mask_img.resize((x1 - x0, y1 - y0), resample=resample)
        np_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        np_mask[y0:y1, x0:x1] = np.array(mask_img)
        masks.append(SegmentationMask(y0, x0, y1, x1, np_mask, item.get("label", "")))
    return masks


def overlay_mask_on_img(img: Image.Image, mask: np.ndarray, color: str, alpha: float = 1.0) -> Image.Image:
    # Prepare RGBA image and overlay layer
    img_rgba = img.convert("RGBA")
    overlay = Image.new("RGBA", img_rgba.size)
    overlay_np = np.zeros((img_rgba.height, img_rgba.width, 4), dtype=np.uint8)
    mask_bool = mask < 50
    # r, g, b = ImageColor.getrgb(color)
    r, g, b = 0, 0, 0
    overlay_np[mask_bool] = (r, g, b, int(alpha * 255))
    overlay_img = Image.fromarray(overlay_np, mode="RGBA")
    return Image.alpha_composite(img_rgba, overlay_img)


def plot_segmentation_masks(img: Image.Image, masks: list[SegmentationMask]) -> Image.Image:
    # Overlay masks and draw bounding boxes with labels
    colors = list(ImageColor.colormap.keys())
    for i, m in enumerate(masks):
        img = overlay_mask_on_img(img, m.mask, colors[i % len(colors)])
    # draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    # for i, m in enumerate(masks):
    #     c = colors[i % len(colors)]
    #     draw.rectangle(((m.x0, m.y0), (m.x1, m.y1)), outline=c, width=4)
    #     if m.label:
    #         draw.text((m.x0 + 8, m.y0 - 20), m.label, fill=c, font=font)
    return img


def main_segmentation():
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    print("loaded Gemini client")
    # MODEL_ID = "gemini-2.5-pro-preview-05-06"
    # MODEL_ID = "gemini-2.5-flash-preview-05-20"
    MODEL_ID = "gemini-2.5-flash-lite-preview-06-17"
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, '..', 'io', 'prompt_gemini.yaml')
    image_path = os.path.join(script_dir, '..', 'image', 'scene2.png')
    # Load prompt
    with open(prompt_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    prompt = cfg["objects"].get("shelf")
    if not prompt:
        raise ValueError("Prompt for segmentation not found in the configuration file.")
    # Load and resize image
    im = Image.open(image_path)
    print(f"Loaded image: {image_path} (size: {im.size})")
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    im.thumbnail((640, 480), resample)
    # Generate segmentation
    if MODEL_ID == "gemini-2.5-flash-lite-preview-06-17":
        thinking_config = types.ThinkingConfig(thinking_budget=512)
    else:
        thinking_config = types.ThinkingConfig(thinking_budget=10)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[prompt, im],
        config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=thinking_config
        )
    )
    print(response)
    # Parse and visualize
    masks = parse_segmentation_masks(response.text, img_height=im.height, img_width=im.width)
    result = plot_segmentation_masks(im, masks).convert('RGB')
    # Save result
    out_path = os.path.join(TARGET_DIR, 'segmentation_result.jpg')
    result.save(out_path, 'JPEG')
    print(f'Saved segmentation result: {out_path}')
    return out_path

if __name__ == '__main__':
    print("Starting segmentation...")
    main_segmentation()

