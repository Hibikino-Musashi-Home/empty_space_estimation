import io
import os
import yaml
import json
from PIL import Image
from flask import Flask, request, jsonify
from google import genai
from google.genai import types
import logging
import re
logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

with open("../io/config.yaml", "r") as f:
    config = yaml.safe_load(f)
model_id = config["MODEL"]["FLASH_20"]

client_gemini = genai.Client(api_key=api_key)

def extract_json(response_text) -> str:
    """
    JSONファイルを抽出する関数
    Args:
        response_text (str): Geminiからのレスポンステキスト
    Returns:
        str: 抽出されたJSON文字列
    """
    try:
        lines = response_text.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                response_text = "\n".join(lines[i+1:])  # Remove everything before "```json"
                response_text = response_text.split("```")[0]  # Remove everything after the closing "```"
                break  # Exit the loop once "```json" is found
        return response_text
    except Exception as e:
        raise ValueError(f"Failed to extract JSON from response: {str(e)}")
    
def get_next_output_index(output_dir, base_name):
        pattern = re.compile(rf"{re.escape(base_name)}_output_(\d+)\.jpg")
        max_index = -1
        for fname in os.listdir(output_dir):
            match = pattern.match(fname)
            if match:
                index = int(match.group(1))
                max_index = max(max_index, index)
        return max_index + 1

@app.route("/empty_space_estimation", methods=["POST"])
def create_chat():
    logging.debug("Received request to /empty_space_estimation")
    files = request.files.getlist("images")
    image = files[0] 
    data = request.form.get("question")
    print(f"Received data: {data}")

    model = "gemini-2.5-flash-lite-preview-06-17"  
    im = Image.open(io.BytesIO(image.read()))
    print(type(im))
    # ThinkingConfigの設定
    if model == "gemini-2.5-flash-lite-preview-06-17":
        thinking_config = types.ThinkingConfig(thinking_budget=512)
    else:
        thinking_config = types.ThinkingConfig(thinking_budget=10)
    response = client_gemini.models.generate_content(
        model=model,
        contents=[
            """
            この画像は棚を正面から見たもので、画像内には複数の番号が記載されています。
            棚の中に物を置く際に、以下の条件をすべて満たすような**適切な番号**を1つだけ選んでください。
            選択した番号の位置が不安定な場合は、番号に移動量（ピクセル数）を併せて指定してください。

            条件：
            - 棚の中であること（棚の外にある番号は絶対に選ばないでください）
            - 落下の危険がない場所（安定して物を置ける棚板の上）
            - 物体が置かれていない場所（他のオブジェクトに近すぎる番号は選ばないでください）
            - カテゴリ（果物, 文房具、調味料、飲み物, ボール)
            - 周囲の物体に同じカテゴリのものがあれば**横(左か右)**の数字を選んでください。
            - 周囲に同じカテゴリの物体がないときは最も広い場所にある番号を選んでください。
            - 同じカテゴリの物体の隣に来るようにoffset_pixelsを指定してください。

            {"recommended_number": <整数>,"offset_pixels": {"x": <整数>,"y": <整数>},"explanation": "<説明文>"}

            注意：
            - 推奨番号は1つだけ選び、他の情報は一切出力しないでください。
            - 条件を満たす番号が存在しない場合は {"recommended_number": null} を出力してください。
            - offset_pixelsは、選択した番号の位置が安定である場合は0を出力してください。
            - 既にある物体はバウンディングボックスで囲まれているので、バウンディングボックスに重なっている番号は選ばないでください。
            """,
            data,
            im
        ],
        config=types.GenerateContentConfig(
        temperature=0.0,
        thinking_config=thinking_config
        )
    )

    reply = response.text
    # 計測終了
    print(f"Geminiの応答: {reply}")
    data = json.loads(reply)
    if "recommended_number" not in data:
        raise ValueError("応答に'recommended_number'が含まれていません。")
    number = data["recommended_number"]
    return str(number)

@app.route("/segmentation", methods=["POST"])
def main_segmentation():
    files = request.files.getlist("images")
    im = files[0]  # 画像ファイルを取得
    # MODEL_ID = "gemini-2.5-pro-preview-05-06"
    # MODEL_ID = "gemini-2.5-flash-preview-05-20"
    MODEL_ID = "gemini-2.5-flash-lite-preview-06-17"
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, '..', 'io', 'prompt_gemini.yaml')
    
    # Load prompt
    with open(prompt_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    prompt = cfg["objects"].get("shelf")
    if not prompt:
        raise ValueError("Prompt for segmentation not found in the configuration file.")
    # Load and resize image

    im = Image.open(io.BytesIO(files[0].read()))
    if MODEL_ID == "gemini-2.5-flash-lite-preview-06-17":
        thinking_config = types.ThinkingConfig(thinking_budget=512)
    else:
        thinking_config = types.ThinkingConfig(thinking_budget=10)
    response = client_gemini.models.generate_content(
        model=MODEL_ID,
        contents=[prompt, im],
        config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=thinking_config
        )
    )
    print(response)
    return response.text

    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)  # Flaskアプリケーションを起動
