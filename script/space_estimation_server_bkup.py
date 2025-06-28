#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import os
import cv2
import re
import numpy as np
import io
from PIL import Image as PILImage
import json
from google import genai
from google.genai import types
import datetime
import glob
import yaml
import dataclasses
import seg_node
import roslib
import requests

from tamlib.utils import Logger
import rospy
from empty_space_estimation.msg import EmptySpace 
from empty_space_estimation.srv import EmptySpaceService, EmptySpaceServiceResponse, EmptySpaceServiceRequest
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge


class Seg2PlaceChatBotGemini:
    """ 棚の写真に対して物を置くのに適した場所を提案するチャットボット """
    def __init__(self):
        ##
        Logger.__init__(self, loglevel="INFO" )
  
        rospy.loginfo("Empty Spaca Estimation Server ready...")
        self.bridge = CvBridge()
        self.srv_estimation = rospy.Service("/empty_space_estimation/service", EmptySpaceService, self.run)
        self.pub_image = rospy.Publisher("/empty_space_estimation/result_image", ROSImage, queue_size=1)
        self.loginfo("Empty space estimation service is ready!")
        


    def add_numbers_to_image(self, image, output_path):
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
        img = self.bridge.compressed_imgmsg_to_cv2(image)
        cv2.imwrite(os.path.join(roslib.packages.get_pkg_dir("empty_space_estimation"), "image/origin_image.jpg"), result_image)
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
        return img

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

    def prepare_file_entry(self, name, value):
        if isinstance(value, str):
            return ('images', (name, open(value, 'rb'), 'image/jpeg'))
        elif isinstance(value, PILImage.Image):
            buffer = io.BytesIO()
            value.save(buffer, format='JPEG')
            buffer.seek(0)
            return ('images', (name, buffer, 'image/jpeg'))
        else:
            raise TypeError(f"Unsupported type for image: {type(value)}")
    
    def run(self):
        req = EmptySpaceServiceRequest()
        image = req.image
        image_path = seg_node.main_segmentation(image)

        output_dir = "../output/seg2place_results"
        select_number = None
        chatbot = Seg2PlaceChatBotGemini()
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        n = self.get_next_output_index(output_dir, base_name)  # 既存画像から番号を決定
        output_path = os.path.join(output_dir, f"{base_name}_output_{n}.jpg")
        image = Image.open(image_path).convert("RGB")
        image = self.add_numbers_to_image(image, output_path)
       

         # Flaskサーバーのエンドポイント
        if rospy.get_param("hsr_type", "exeception") != "exeception":
            url = "http://192.168.0.10:5001/empty_space_estimation"
        else:
            url = "http://localhost:5001/empty_space_estimation"
        print(f"Using URL: {url}")

        files = [self.prepare_file_entry("image", image)]

        self.loginfo("Sending request to the server...")
        select_number = requests.post(url, files=files)
        self.loginfo(f"Response from server: {select_number}")
        image = self.add_number_to_image_next(select_number, image, output_path)
        


if __name__ == "__main__":
    chatbot = Seg2PlaceChatBotGemini()
    chatbot.run()

