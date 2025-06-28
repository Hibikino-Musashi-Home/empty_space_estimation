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
import datetime
import glob
import yaml
import dataclasses
import seg_node
import roslib
import requests
import subprocess

from tamlib.utils import Logger
import rospy
from empty_space_estimation.msg import EmptySpace 
from empty_space_estimation.srv import EmptySpaceService, EmptySpaceServiceResponse, EmptySpaceServiceRequest

from sensor_msgs.msg import Image as ROSImage, PointCloud2
from sensor_msgs import point_cloud2 as pc2
from cv_bridge import CvBridge
import math
from visualization_msgs.msg import Marker


class Seg2PlaceChatBotGemini:
    """ 棚の写真に対して物を置くのに適した場所を提案するチャットボット """
    def __init__(self):
        ##
        Logger.__init__(self, loglevel="INFO" )
        rospy.init_node('empty_space_estimation_server', anonymous=True)
        rospy.loginfo("Empty Spaca Estimation Server ready...")
        self.bridge = CvBridge()
        self.srv_estimation = rospy.Service("/empty_space_estimation/service", EmptySpaceService, self.run)
        self.pub_image = rospy.Publisher("/empty_space_estimation/result_image", ROSImage, queue_size=1)
        rospy.loginfo("Empty space estimation service is ready!")
        self.marker_pub = rospy.Publisher('selected_point_marker', Marker, queue_size=1, latch=True)
        


    def add_numbers_to_image(self, img, output_path):
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
        height, width = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 0, 0)
        thickness = 1

        num_rows = 12
        num_cols = 12
        cell_height = height // num_rows
        cell_width = width // num_cols
        self.number_position = []
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
        return output_path

    def get_next_output_index(self, output_dir, base_name):
        pattern = re.compile(rf"{re.escape(base_name)}_output_(\d+)\.jpg")
        max_index = -1
        for fname in os.listdir(output_dir):
            match = pattern.match(fname)
            if match:
                index = int(match.group(1))
                max_index = max(max_index, index)
        return max_index + 1

    

    def add_number_to_image_next(self, select_number, output_path):
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
        return center_x, center_y


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
        
    def save_image(self, ros_image_msg):
        package_path = roslib.packages.get_pkg_dir("empty_space_estimation")

        # パッケージパスの後ろに続けるパス
        file_path = os.path.join(package_path, "io", "config.yaml")
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            rospy.loginfo(f"Config loaded: {config}")
        input_path = config["PATH"]["IMG_TARGET"]
        if ros_image_msg.encoding == "8UC3":
            ros_image_msg.encoding = "bgr8"
        # ROS Imageメッセージ → OpenCV形式に変換（BGR8）
        cv_image = self.bridge.imgmsg_to_cv2(ros_image_msg, desired_encoding='bgr8')

        # JPEGで保存
        cv2.imwrite(input_path, cv_image)

        print(f"Saved image to {input_path}")

        rospy.loginfo(f"Saving image to {input_path}")

        return input_path
        
    def get_3d_point_from_pixel_correct_nan(self, pc_msg, x, y):
        width = pc_msg.width
        height = pc_msg.height
        rospy.loginfo(f"PointCloud2 width: {width}, height: {height}, start x: {x}, y: {y}")

        start_index = y * width + x
        gen = pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=False)
        points = list(enumerate(gen))

        if start_index >= len(points):
            rospy.logwarn("Start index out of range.")
            return None

        _, first_pt = points[start_index]
        x1, y1, z1 = first_pt

        # 最初がNaNでなければそのまま返す
        if not (math.isnan(x1) or math.isnan(y1) or math.isnan(z1)):
            rospy.loginfo(f"Start point is valid: {first_pt}")
            return first_pt

        # 最初がNaNの場合のみ右に探索
        valid_points = []
        valid_dx = []  # 有効点のdxを保存
        nan_streak = 0

        for dx in range(1, width - x):
            idx = start_index + dx
            if idx >= len(points):
                rospy.logwarn("Index out of range during search.")
                break

            _, pt = points[idx]
            x_val, y_val, z_val = pt

            if math.isnan(x_val) or math.isnan(y_val) or math.isnan(z_val):
                nan_streak += 1
                valid_points.clear()
                valid_dx.clear()
                rospy.loginfo(f"NaN at dx={dx}, resetting valid_points")
                if nan_streak >= 2:
                    rospy.loginfo("Two consecutive NaNs encountered, stopping search.")
                    break
            else:
                nan_streak = 0
                valid_points.append(pt)
                valid_dx.append(dx)
                rospy.loginfo(f"Valid point at dx={dx}: {pt}")
                if len(valid_points) >= 2:
                    break

        if len(valid_points) < 2:
            rospy.logwarn("Could not find two consecutive valid points for correction.")
            return None

        # 1ピクセルあたりの距離を計算（2連続有効点のx座標差 / dx差）
        x_a, y_a, z_a = valid_points[0]
        x_b, y_b, z_b = valid_points[1]
        dx_a = valid_dx[0]
        dx_b = valid_dx[1]

        pixel_distance = (x_b - x_a) / (dx_b - dx_a)
        rospy.loginfo(f"Pixel distance per 1 pixel: {pixel_distance}")

        # 最後の有効点dxから、最初のNaN位置（dx=0）までの距離補正を計算
        # ここでは最後の有効点のx座標から、移動距離分(pixel_distance * dx)を掛けて補正
        total_shift = pixel_distance * dx_b
        corrected_x = x_b - total_shift
        rospy.loginfo(f"Corrected x coordinate: {corrected_x}")

        # y,zは最後の有効点の値を使う
        corrected_pt = (corrected_x, y_b, z_b)

        rospy.loginfo(f"Corrected point: {corrected_pt}")
        return corrected_pt

    
    def run(self, req):
        image = req.image
        input_image_path = self.save_image(image)
        
        # image_path = seg_node.main_segmentation(input_image_path)

        select_number = None
        package_path = roslib.packages.get_pkg_dir("empty_space_estimation")

        # パッケージパスの後ろに続けるパス
        file_path = os.path.join(package_path, "io", "config.yaml")
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            rospy.loginfo(f"Config loaded: {config}")
        add_number_path = config["PATH"]["IMG_ADD_NUMBERS"]
        

    
        image = cv2.imread(input_image_path)
        image = self.add_numbers_to_image(image, add_number_path)
       

        # Flaskサーバーのエンドポイント
        if rospy.get_param("hsr_type", "exeception") != "exeception":
            url = "http://192.168.0.10:5001/empty_space_estimation"
        else:
            url = "http://localhost:5001/empty_space_estimation"
        print(f"Using URL: {url}")

        files = [self.prepare_file_entry("image", image)]

        rospy.loginfo("Sending request to the server...")
        select_number = requests.post(url, files=files)
        rospy.loginfo(f"Response from server: {select_number.text}")
                # パッケージパスの後ろに続けるパス
        file_path = os.path.join(package_path, "io", "config.yaml")
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            rospy.loginfo(f"Config loaded: {config}")
        add_marker_path = config["PATH"]["IMG_ADD_NUMBERS"]
        x, y = self.add_number_to_image_next(select_number.text, add_marker_path)
        
        response = EmptySpaceServiceResponse()
        rospy.loginfo(req.point.header.frame_id)
        rospy.loginfo(f"Received point: {req.point}")
        response.results.frame_id = str(req.point.header.frame_id)
        rospy.loginfo(f"Frame ID: {response.results.frame_id}")
        

      
        point = self.get_3d_point_from_pixel_correct_nan(req.point, x, y)
        x, y, z = point
        response.results.x = x
        response.results.y = y

       
        marker = Marker()
        marker.header.frame_id = str(req.point.header.frame_id)  # ここはあなたの点群のフレームに合わせてください
        marker.header.stamp = rospy.Time.now()
        marker.ns = str(req.point.header.frame_id)
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z

        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime = rospy.Duration()

        self.marker_pub.publish(marker)
                
        return response
        


if __name__ == "__main__":
    chatbot = Seg2PlaceChatBotGemini()
    rospy.spin()

