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
import empty_space_estimation.seg_node as seg_node
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
        self.vizualization = rospy.get_param("~visualize", False)
        fx, fy, cx, cy = rospy.get_param("/hsrb/head_rgbd_sensor/depth_registered/camera_info", [525.0, 525.0, 319.5, 239.5])
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        if self.vizualization:
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
            return
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # 処理時間だけを抽出して平均を計算
        times = [entry["処理時間"] for entry in data]
        average_time = sum(times) / len(times)

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
        
        """ 補完済み深度マップからの 3D 点を計算して返す """
        z = float(self.filled_z[x, y])
        # ピクセル→実空間の変換
        x = (x - self.cx) * z / self.fx
        y = (y - self.cy) * z / self.fy

        return x, y, z

    def build_filled_depth_map(self, pc_msg):
        """ 点群から Z マップを作り、NaN 部分をインペイントして返す """
        width = pc_msg.width
        height = pc_msg.height

        # 点群から Z 値だけ抜き出して 2D マップに
        z_list = [pt[2] for pt in pc2.read_points(pc_msg, field_names=("x","y","z"), skip_nans=False)]
        z_map = np.array(z_list, dtype=np.float32).reshape((height, width))

        # NaN 部分をマスク化
        nan_mask = np.isnan(z_map).astype(np.uint8)

        # インペイント（NS 法）
        z_filled = cv2.inpaint(z_map, nan_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
        return z_filled

    
    def run(self, req):
        if not hasattr(self, '_z_filled'):
            self.filled_z = self.build_filled_depth_map(req.point)

        self.cvbridge = CvBridge()
        image = req.image
        
        # image_path = seg_node.main_segmentation(input_image_path)

        select_number = None
        package_path = roslib.packages.get_pkg_dir("empty_space_estimation")

        # パッケージパスの後ろに続けるパス
        file_path = os.path.join(package_path, "io", "config.yaml")
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        add_number_path = config["PATH"]["IMG_ADD_NUMBERS"]
        

    
        image = self.cvbridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        image = self.add_numbers_to_image(image, add_number_path)
       

        # Flaskサーバーのエンドポイント
        if rospy.get_param("hsr_type", "exeception") != "exeception":
            url = "http://192.168.0.10:5001/empty_space_estimation"
        else:
            url = "http://localhost:5001/empty_space_estimation"
        print(f"Using URL: {url}")

        files = [self.prepare_file_entry("image", image)]
        # 質問の用意\
        data = {"question": str(req.question)}

        rospy.loginfo("Sending request to the server...")
        select_number = requests.post(url, files=files, data=data)
        rospy.loginfo(f"Response from server: {select_number.text}")
                # パッケージパスの後ろに続けるパス
        file_path = os.path.join(package_path, "io", "config.yaml")
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        add_marker_path = config["PATH"]["IMG_ADD_NUMBERS"]
        x, y = self.add_number_to_image_next(select_number.text, add_marker_path)
        
        response = EmptySpaceServiceResponse()
        response.results.frame_id = str(req.point.header.frame_id)
        rospy.loginfo(f"Frame ID: {response.results.frame_id}")
       


      
        point = self.get_3d_point_from_pixel_correct_nan(req.point, x, y)
        x, y, z = point
        response.results.x = x
        response.results.y = y

        if self.vizualization:
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

            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05

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

