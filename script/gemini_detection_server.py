#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import re
import yaml
import json
import actionlib
import rospy
import requests
from PIL import Image as PILImage, ImageDraw, ImageColor
from tam_gemini_detection.msg import BBox, ObjectDetection, Pose3D
from tam_gemini_detection.msg import GeminiDetectionAction, GeminiDetectionResult
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2
import numpy as np


class GeminiDetectionServer:
    def __init__(self):
        rospy.loginfo("Gemini Detection Server ready...")
        self.bridge = CvBridge()

        # Load configuration from YAML file
        config_path = "/home/hma/ros_ws/src/5_skills/tam_gemini_detection/io/config/config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        # self.where = config["WHERE"]
        self.model_id = config["MODEL"]["FLASH_20"]
        self.dir_img_target = config["DIR"]["IMG_TARGET"]
        self.dir_img_detected = config["DIR"]["IMG_DETECTED"]
        self.dir_yaml = config["DIR"]["YAML"]
        self.path_base_img = config["PATH"]["BASE_IMG"]
        self.path_comp_img = config["PATH"]["COMP_IMG"]
        self.path_detect_img = config["PATH"]["DETECT_IMG"]
        self.path_base_yaml = config["PATH"]["BASE_YAML"]
        self.path_comp_yaml = config["PATH"]["COMP_YAML"]
        
        self.as_gemini_detection = actionlib.SimpleActionServer("gemini_detection_action", GeminiDetectionAction, self.main)
        self.as_gemini_detection.start()

        self.pub_image = rospy.Publisher("/gemini_detection_action/result_image", ROSImage, queue_size=1)
        self.logsuccess("ready to gemini detection")


    def extract_json(self, json_output):
        """
        JSONファイルを抽出する関数

        Args:
            json_output: geminiからの出力

        Returns:
            json_output: 抽出されたJSON形式の文字列
        """
        # Parsing out the markdown fencing
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
                json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
                break  # Exit the loop once "```json" is found
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

    
    def save_yaml_to_file(self, yaml_output, check, where) -> None:
        """
        YAMLファイルを保存する関数
        Args:
            yaml_output: geminiからの出力
            check: "detection" or "detection_base"
            where: ロボットの位置情報（例：living room、dining roomなど）
        """
        if isinstance(yaml_output, str):
            data = yaml.safe_load(yaml_output)
        else:
            data = yaml_output
        
        # 保存先ファイル名の設定
        if (check == "detection"):
            file_name = os.path.join(self.path_comp_yaml + where + ".yaml")
        elif (check == "detection_base"):
            file_name = os.path.join(self.path_base_yaml + where + ".yaml")
        else:
            return 0
        file_path = os.path.join(self.dir_yaml, file_name)

        # ファイルが存在する場合は追記，存在しない場合は新規作成
        if os.path.exists(file_path):
            with open(file_path, mode='a', encoding="utf-8") as f:
                f.write("\n---\n")
                yaml.safe_dump(data, f, indent=2)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, indent=2)

    def plot_save_bounding_boxes(self, im, bounding_boxes, where, is_base) -> PILImage:
        """
        BBOXを画像に描画する関数
        Args:
            im: PIL Imageオブジェクト
            bounding_boxes: YAML形式のバウンディングボックス情報
            where: ロボットの位置情報（例：living room、dining roomなど）
            is_base: 基準画像かどうか（Trueなら基準画像、Falseなら比較画像）
        Returns:
            img: 描画された画像のPIL Imageオブジェクト
        """
        # Load the image
        img = im
        width, height = img.size
        print(img.size)
        # Create a drawing object
        draw = ImageDraw.Draw(img)
        # add colors
        additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

        # Define a list of colors
        colors = [
        'red',
        'green',
        'blue',
        'yellow',
        'orange',
        'pink',
        'purple',
        'brown',
        'gray',
        'beige',
        'turquoise',
        'cyan',
        'magenta',
        'lime',
        'navy',
        'maroon',
        'teal',
        'olive',
        'coral',
        'lavender',
        'violet',
        'gold',
        'silver',
        ] + additional_colors

        # Parsing out the markdown fencing
        # print(bounding_boxes)
        # bounding_boxes = extract_yaml(bounding_boxes)

        # Load the bounding boxes from YAML format
        bounding_boxes = yaml.safe_load(bounding_boxes)

        # Iterate over the bounding boxes
        for i, bounding_box in enumerate(bounding_boxes):
            print(bounding_box)
            print(bounding_box["box_2d"])
            # Select a color from the list
            color = colors[i % len(colors)]

            # Convert normalized coordinates to absolute coordinates
            abs_y_min = int(bounding_box["box_2d"][0]/1000 * height)
            abs_x_min = int(bounding_box["box_2d"][1]/1000 * width)
            abs_y_max = int(bounding_box["box_2d"][2]/1000 * height)
            abs_x_max = int(bounding_box["box_2d"][3]/1000 * width)

            if abs_x_min > abs_x_max:
                abs_x_min, abs_x_max = abs_x_max, abs_x_min

            if abs_y_min > abs_y_max:
                abs_y_min, abs_y_max = abs_y_max, abs_y_min

            # Draw the bounding box
            draw.rectangle(
                ((abs_x_min, abs_y_min), (abs_x_max, abs_y_max)), outline=color, width=4
            )

            # Draw the text
            if "label" in bounding_box:
                draw.text((abs_x_min + 8, abs_y_min + 6), bounding_box["label"], fill=color)

        # Save the image with bounding boxes
        img = img.convert("RGB")
        if is_base:
            index = self.get_next_image_index(where, is_base=True)
            filename = f"{self.path_detect_img}{where}_{index}.jpg"
        else:
            index = self.get_next_image_index(where, is_base=False)
            filename = f"{self.path_comp_img}{where}_{index}.jpg"

        filepath = os.path.join(self.dir_img_detected, filename)
        print(f"Saving image to {filepath}")
        img.save(filepath, "JPEG")

        return img

    def get_next_image_index(self, where, is_base) -> int:
        """
        compared_image_<index>.jpg, detected_image_<index>.jpgの最大のindex + 1を返します。
        Args:
            where: ロボットの位置情報（例：living room、dining roomなど）
            is_base: 基準画像かどうか（Trueなら基準画像、Falseなら比較画像）
        Returns:
            int: 次の比較画像のインデックス
        """

        # Create directory if it doesn't exist
        os.makedirs(self.dir_img_detected, exist_ok=True)

        # Define the pattern for detected image files
        if is_base:
            pattern = re.compile(f'^{self.path_detect_img}{where}_(\d+)\.jpg$')
        else:
            pattern = re.compile(f'^{self.path_comp_img}{where}_(\d+)\.jpg$')
            
        max_index = 0
        # List files in the current directory
        try:
            for fname in os.listdir(self.dir_img_detected):
                m = pattern.match(fname)
                if m:
                    # Extract the index and update max_index if necessary
                    index = int(m.group(1))
                    if index > max_index:
                        max_index = index
        except FileNotFoundError:
            # Directory doesn't exist yet, start with index 1
            pass # max_index remains 0
                        
        return max_index + 1

    def compare_image(self, base_image, compare_image, base_yaml_path, where,index=None):
        """
        画像を比較して新しいオブジェクトを検出する関数
        Args:
            base_image: 基準画像のパス，またはPIL Imageオブジェクト
            compare_image: 比較画像のパス，またはPIL Imageオブジェクト，またはROSImage(sensor_msgs Image)オブジェクト
            base_yaml_path: 基準画像のBounding boxを保存したYAMLファイルのパス．複数回分の検出結果が保存されている場合は、最新のものを使用する処理になっている
            where: ロボットの位置情報（例：living room、dining roomなど）
            index: オプションで比較画像のインデックス
        Returns:
            response_parsed: 検出結果のYAML形式の文字列
            image_response: 検出結果の画像のPIL Imageオブジェクト
        """
        # 基準画像の読み込み
        # ファイル入力の場合
        if base_image is str and os.path.exists(base_image): 
            image_base = PILImage.open(base_image)
        # 画像が直接入力された場合
        else: 
            image_base = base_image

        # 比較画像の読み込み
        # ファイル入力の場合
        # if compare_image is str and os.path.exists(compare_image):
        #     image_compare = PILImage.open(compare_image)
        # # 画像が直接入力された場合
        # else:
        #     image_compare = compare_image
            
        # 比較画像の読み込み
        if isinstance(compare_image, str) and os.path.exists(compare_image):
            image_compare = PILImage.open(compare_image)
        elif isinstance(compare_image, ROSImage):
            cv_image = self.bridge.imgmsg_to_cv2(compare_image, desired_encoding='bgr8')
            image_compare = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        elif isinstance(compare_image, PILImage.Image):
            image_compare = compare_image
        else:
            raise TypeError(f"Unsupported type for compare_image: {type(compare_image)}")

        # YAMLファイルの読み込み
        with open(base_yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load_all(f)
            #  一番最新のYAMLデータのみを使用する
            for doc in yaml_data:
                print(doc)
                yaml_text = yaml.safe_dump(doc)
            
        print(yaml_text)

        # Convert the YAML string to JSON format
        # ym = yaml.safe_load(yaml_text)
        # image_response_0_json = json.dumps(ym, indent=2)

        # Flaskサーバーのエンドポイント
        if rospy.get_param("hsr_type", "exeception") != "exeception":
            url = "http://192.168.0.10:5001/ask_gemini"
        else:
            url = "http://localhost:5001/ask_gemini"
        
        print(f"URL: {url}")

        files = [
            self.prepare_file_entry("base_image", image_base),
            self.prepare_file_entry("compare_image", image_compare),
        ]

        data = {
            'yaml_data': yaml_text if isinstance(yaml_text, str) else yaml.safe_dump(yaml_text),
            'is_base': False, # 基準画像ではない string型で送られることに注意
        }

        # # data(str) -> data_yaml(YAML)
        # if isinstance(data, str):
        #     data = yaml.safe_load(data)

        # check the is_base
        print(f"Is base image: {data['is_base']}")
        print(f"Data type: {type(data)}")
        print(f"files: {files}")
        # POSTリクエストを送信
        response = requests.post(url, files=files, data=data)
        response_yaml = response.text
        print(type(response))

        # Print the coordinates
        print(response_yaml)
        
        self.save_yaml_to_file(response_yaml, "detection", where)
        
        # Add bounding boxes to the image
        image_response = self.plot_save_bounding_boxes(image_compare, response_yaml, where, is_base=False)

        return response_yaml, image_response

    def main(self, goal):
        rospy.loginfo("Received goal for Gemini detection")
        where = goal.where
        base_image = os.path.join(self.dir_img_target, self.path_base_img + where + ".jpg")
        base_yaml = os.path.join(self.dir_yaml, self.path_base_yaml + where + ".yaml")
       
        yaml_response_parsed, image_response = self.compare_image(base_image, goal.image, base_yaml, where)
        
        # 検出結果の報告
        result = GeminiDetectionResult()
        image_response_parsed_yaml = yaml.safe_load(yaml_response_parsed)
        print("--- Report of Detection ---")
        print(image_response_parsed_yaml)

        added_frame_bool = False
        for data in image_response_parsed_yaml:
            print(data)
            if data["added_frame"] == True:
                print("--- Object Detected ---")
                print(f"Object : {data['label']}")
                print(f"Bounding Box : {data['box_2d']}")
                added_frame_bool = True
                result.is_detected = True
                result.bbox.name = data['label']
                result.bbox.x = data['box_2d'][1] # x_min
                result.bbox.y = data['box_2d'][0] # y_min
                result.bbox.w = data['box_2d'][3] - data['box_2d'][1] # x_max - x_min
                result.bbox.h = data['box_2d'][2] - data['box_2d'][0] # y_max - y_min
                break
            else:
                pass

        if (added_frame_bool == False):
            print("--- No new objects detected ---")
            result.is_detected = False
            result.bbox.name = ""
            result.bbox.x = -1
            result.bbox.y = -1
            result.bbox.w = -1
            result.bbox.h = -1

        self.as_gemini_detection.set_succeeded(result)
        ros_image = self.bridge.cv2_to_imgmsg(cv2.cvtColor(np.array(image_response), cv2.COLOR_RGB2BGR), encoding="bgr8")
        self.pub_image.publish(ros_image)

    def delete(self):
        rospy.loginfo("Shutting down")
        return

if __name__ == "__main__":
    rospy.init_node("gemini_detection_server")

    cls = GeminiDetectionServer()
    rospy.on_shutdown(cls.delete)
    while not rospy.is_shutdown():
        try:
            pass 
        except rospy.exceptions.ROSException:
            rospy.logerr("[" + rospy.get_name() + "]: FAILURE")
        rospy.sleep(1)