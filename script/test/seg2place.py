import base64
import os
import cv2
import re
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageColor
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
import datetime
import glob
import yaml
import dataclasses
import script.test.seg as seg


class Seg2PlaceChatBotGemini:
    """ 棚の写真に対して物を置くのに適した場所を提案するチャットボット """
    def __init__(self):
        load_dotenv()
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = "gemini-2.5-flash-preview-05-20"
        # self.model = "gemini-2.5-pro-preview-05-06"
        # self.model = "gemini-2.5-flash-lite-preview-06-17"
        self.n = 0
        self.number_position = []
        self.input_image_path = "../image/scene1.jpg"
        self.output_dir = "../output/seg2place_results"

    def add_numbers_to_image(self, image_path, output_path):
        # img = cv2.imread(image_path)
        # height, width = img.shape[:2]
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 1
        # font_color = (0, 0, 0)
        # thickness = 2

        # num_rows = 8
        # num_cols = 8
        # cell_height = height // num_rows
        # cell_width = width // num_cols

        # number = 0
        # for row in range(num_rows):
        #     for col in range(num_cols):
        #         x = col * cell_width + cell_width // 2
        #         y = row * cell_height + cell_height // 2
        #         text = str(number)
        #         text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        #         text_x = x - text_size[0] // 2
        #         text_y = y + text_size[1] // 2
        #         self.number_position.append((x, y))
        #         cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, thickness)
        #         number += 1

        # cv2.imwrite(output_path, img)
        # print(f"数字を追加した画像を {output_path} に保存しました。")

        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 0, 0)
        thickness = 1

        num_rows = 12
        num_cols = 12
        cell_height = height // num_rows
        cell_width = width // num_cols

        number = 0
        for row in range(num_rows):
            for col in range(num_cols):
                x = col * cell_width + cell_width // 2
                y = row * cell_height + cell_height // 2
                text = str(number)
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = x - text_size[0] // 2
                text_y = y + text_size[1] // 2
                self.number_position.append((x, y))
                cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, thickness)
                number += 1

        cv2.imwrite(output_path, img)
        print(f"数字を追加した画像を {output_path} に保存しました。")

    def get_next_output_index(self, output_dir, base_name):
        pattern = re.compile(rf"{re.escape(base_name)}_output_(\d+)\.jpg")
        max_index = -1
        for fname in os.listdir(output_dir):
            match = pattern.match(fname)
            if match:
                index = int(match.group(1))
                max_index = max(max_index, index)
        return max_index + 1

    

    def add_number_to_image_next(self, select_number, image_path, output_path):
        img = cv2.imread(output_path)
        center_x, center_y = self.number_position[int(select_number)]
        marker_color = (255, 0, 0)
        marker_size = 40
        top_left = (center_x - marker_size // 2, center_y - marker_size // 2)
        bottom_right = (center_x + marker_size // 2, center_y + marker_size // 2)
        cv2.rectangle(img, top_left, bottom_right, marker_color, 2)
        cv2.imwrite(output_path, img)
        print(f"選択された数字：{select_number}")
        print(f"マーカーを追加した画像を {output_path} に保存しました。")

    def encode_image(self, image_path) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def record_timt2yaml(self, start_time, end_time):
        output_path = "../io/processing_time.yaml"
        elapsed_sec = end_time - start_time

        # ① 既存のデータを読み込んでリスト化
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or []
                # 辞書だったらリストにラップ
                if isinstance(data, dict):
                    data = [data]
        else:
            data = []

        # ② 新しいエントリを追加（秒数のみ）
        data.append({"処理時間": round(elapsed_sec.total_seconds(), 6)})

        # ③ 全データを YAML シーケンスで上書き保存
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True)

        print(f"処理時間: {elapsed_sec.total_seconds():.6f} 秒を {output_path} に記録しました。")

        # ④ 平均計算
        self.calculate_average_elapsed_time(output_path)
        
        return 

    def calculate_average_elapsed_time(self, yaml_path):
        if not os.path.exists(yaml_path):
            print(f"ファイルが見つかりません: {yaml_path}")
            return
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # 処理時間だけを抽出して平均を計算
        times = [entry["処理時間"] for entry in data]
        average_time = sum(times) / len(times)

        print(f"平均処理時間: {average_time:.6f} 秒")
        return


    def extract_json(self, json_output: str) -> str:
        # Remove markdown fencing to extract raw JSON
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                json_output = "\n".join(lines[i+1:])
                json_output = json_output.split("```")[0]
                break
        return json_output

    def create_chat(self, select_number, image_path, output_dir):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        n = self.get_next_output_index(output_dir, base_name)  # 既存画像から番号を決定
        output_path = os.path.join(output_dir, f"{base_name}_output_{n}.jpg")

        if self.n == 0:
            self.add_numbers_to_image(image_path, output_path)
        else:
            self.add_number_to_image_next(select_number, image_path, output_path)

        im = Image.open(output_path).convert("RGB")
        question = "リンゴを置くのにふさわしい場所はどこですか？"
        start_time = datetime.datetime.now()  # 計測開始
        print(f"start_processing: {start_time}")
        # ThinkingConfigの設定
        if self.model == "gemini-2.5-flash-lite-preview-06-17":
            thinking_config = types.ThinkingConfig(thinking_budget=512)
        else:
            thinking_config = types.ThinkingConfig(thinking_budget=10)
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                """
                この画像は棚を正面から見たもので、画像内には複数の番号が記載されています。
                棚の中に物を置く際に、以下の条件をすべて満たすような**適切な番号**を1つだけ選んでください。
                選択した番号の位置が不安定な場合は、番号に移動量（ピクセル数）を併せて指定してください。

                条件：
                - 棚の中であること（棚の外にある番号は絶対に選ばないでください）
                - 落下の危険がない場所（安定して物を置ける棚板の上）
                - カテゴリ（果物, 文房具、調味料、飲み物, ボール)
                - 周囲の物体に同じカテゴリのものがあればその横の数字を選んでください。
                - 周囲に同じカテゴリの物体がないときは最も広い場所にある番号を選んでください。
                - 空きスペースにある番号を優先してください（物体がすでにある番号は除外）
                - 同じカテゴリの物体の近くに来るようにoffset_pixelsを指定してください。

                {"recommended_number": <整数>,"offset_pixels": {"x": <整数>,"y": <整数>},"explanation": "<説明文>"}

                注意：
                - 推奨番号は1つだけ選び、他の情報は一切出力しないでください。
                - 条件を満たす番号が存在しない場合は {"recommended_number": null} を出力してください。
                - offset_pixelsは、選択した番号の位置が安定である場合は0を出力してください。
                - 既にある物体はバウンディングボックスで囲まれているので、バウンディングボックスに重なっている番号は選ばないでください。
                """,
                question,
                im
            ],
            config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=thinking_config
            )
        )

        reply = response.text
        # 計測終了
        end_time = datetime.datetime.now()
        self.record_timt2yaml(start_time, end_time)
        print(f"Geminiの応答: {reply}")
        data = json.loads(reply)
        if "recommended_number" not in data:
            raise ValueError("応答に'recommended_number'が含まれていません。")
        number = data["recommended_number"]
        self.add_number_to_image_next(number, image_path, output_path)
        return reply
    
    def run(self):
        input_image_path = "../image/scene1.jpg"
        output_dir = "../output/seg2place_results"
        select_number = None
        chatbot = Seg2PlaceChatBotGemini()
        print("Starting Seg2PlaceChatBotGemini...")
        os.makedirs(output_dir, exist_ok=True)
        response = chatbot.create_chat(select_number, input_image_path, output_dir)
        print(f"Response: {response}")


if __name__ == "__main__":
    input_image_path = seg.main_segmentation()  # Run segmentation test first

    chatbot = Seg2PlaceChatBotGemini()
    TEST_RUNS = 1
    print("Starting test runs...")
    output_dir = "../output/seg2place_results"
    select_number = None
    os.makedirs(output_dir, exist_ok=True)
    response = chatbot.create_chat(select_number, input_image_path, output_dir)

