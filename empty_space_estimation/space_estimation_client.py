#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, PointCloud2, CompressedImage
from empty_space_estimation.srv import EmptySpaceService, EmptySpaceServiceRequest
from nanosam_detection.srv import ObjectDetectionService, ObjectDetectionServiceRequest
import cv2
import os
from cv_bridge import CvBridge
import roslib
import numpy as np
from torch import Tensor
import torch
from typing import Sequence
from empty_space_estimation import seg_node
import yaml

class SpaceEstimationClient:
    def __init__(self):
        rospy.init_node("empty_space_client")
        self.bridge = CvBridge()

        rospy.wait_for_service("/empty_space_estimation/service")


        # ServiceProxy登録
        self.client = rospy.ServiceProxy("/empty_space_estimation/service", EmptySpaceService)
        self.client_nanosam = rospy.ServiceProxy("/object_detection/service", ObjectDetectionService)
        rospy.loginfo("Empty Space Estimation Client initialized.")

    def callback(self, msg):
        self.image_msg = msg  # ← ここで画像を保存

    def mark_image(
        self,
        image: np.ndarray,
        bboxes,
    ) -> np.ndarray:
        """結果の可視化
        Args:
            image (np.ndarray): 入力画像．
            bbox: BBox情報．
        Returns:
            np.ndarray: 描画画像．
        """
        if image.encoding == "8UC3":
            image.encoding = "bgr8"
        result_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')


        for box in bboxes:
            x_min = int(box.x - box.w / 2)
            y_min = int(box.y - box.h / 2)
            x_max = int(box.x + box.w / 2)
            y_max = int(box.y + box.h / 2)

            # BBoxを描画
            cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # ラベルテキスト
            text = str(box.name)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 1
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

            # ラベルの描画位置
            text_x = x_min
            text_y = y_min - 5 if y_min - 5 > text_size[1] else y_min + text_size[1] + 5

        #     # 背景矩形を描画（文字の下に白背景）
        #     cv2.rectangle(
        #       result_image,
        #       (text_x, text_y - text_size[1] - 2),
        #       (text_x + text_size[0] + 4, text_y + 2),
        #       (255, 255, 255),  # 白背景
        #       thickness=cv2.FILLED
        #     )

        #    # テキストを描画（赤文字）
        #     cv2.putText(
        #       result_image,
        #       text,
        #       (text_x + 2, text_y),
        #       font,
        #       font_scale,
        #       (0, 0, 255),  # 赤色
        #       thickness,
        #       cv2.LINE_AA
        #     )
        return result_image
    
    def save_image(self, ros_image_msg):
        package_path = roslib.packages.get_pkg_dir("empty_space_estimation")
        # パッケージパスの後ろに続けるパス
        file_path = os.path.join(package_path, "io", "config.yaml")
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        # 画像の保存先パスを取得
        if self.n == 0:
            input_path = config["PATH"]["IMG_TARGET"]
            self.n += 1
        elif self.n == 1:
            input_path = config["PATH"]["IMG_BBOX"]
            self.n += 1
        else:
            input_path = config["PATH"]["IMG_MULTI"]
        rospy.loginfo(type(ros_image_msg))
        if isinstance(ros_image_msg, Image):
            rospy.loginfo(type(ros_image_msg))
            if ros_image_msg.encoding == "8UC3":
                ros_image_msg.encoding = "bgr8"
                cv_image = self.bridge.imgmsg_to_cv2(ros_image_msg, desired_encoding='bgr8')
            else:
                cv_image = self.bridge.imgmsg_to_cv2(ros_image_msg, desired_encoding=ros_image_msg.encoding)

        elif isinstance(ros_image_msg, CompressedImage):
            cv_image = self.bridge.compressed_imgmsg_to_cv2(ros_image_msg, desired_encoding='bgr8')
        
        else:
            cv_image = ros_image_msg
            
        # ROS Imageメッセージ → OpenCV形式に変換（BGR8）
       
        # JPEGで保存
        cv2.imwrite(input_path, cv_image)

        rospy.loginfo(f"Saving image to {input_path}")

        return input_path
    

    def run(self):
        self.n = 0
        try:
            # nanosamを呼び出す
            package = roslib.packages.get_pkg_dir("empty_space_estimation")
            ymal_path = os.path.join(package, "io", "nanosam.yaml")
            point = rospy.wait_for_message("/hma_pcl_reconst/depth_registered/points", PointCloud2, timeout=10)
            request_nanosam = ObjectDetectionServiceRequest(use_latest_image=True, specific_id=ymal_path, max_distance=1.0)
            response_nanosam = self.client_nanosam(request_nanosam)
            input_image_path = self.save_image(response_nanosam.detections.rgb)
            
            # bboxのみの画像を保存
            bbox_image = self.mark_image(self.bridge.cv2_to_imgmsg(cv2.imread(input_image_path), encoding="bgr8"), response_nanosam.detections.bbox)
            self.save_image(bbox_image)

            image_path = seg_node.main_segmentation(input_image_path)
            use_image = self.bridge.cv2_to_imgmsg(cv2.imread(image_path), encoding="bgr8")

            
            detection_image = self.mark_image(use_image, response_nanosam.detections.bbox)
            self.save_image(detection_image)
            
            

            # 空き領域推定サービス呼び出し
            request = EmptySpaceServiceRequest()
            # #  画像を保存
            # package_path = roslib.packages.get_pkg_dir("empty_space_estimation")
            # file_path = os.path.join(package_path, "input", "tmp_input_image.jpg")
            # self.save_image(self.image_msg, file_path)
            target_obj = rospy.get_param("/target_obj", "コップ")
            request.question = f"棚の画像を解析して、{target_obj}置くのに適した場所を提案してください。"
            request.image = self.bridge.cv2_to_imgmsg(detection_image, encoding="bgr8")
            request.point = point  # PointCloud2メッセージをセット
            response = self.client(request)

            rospy.loginfo("空き領域推定結果:")
            rospy.loginfo(response.results)
            return response.results

        except rospy.ServiceException as e:
            rospy.logerr("サービス呼び出しに失敗しました: %s", str(e))


if __name__ == "__main__":
    cls = SpaceEstimationClient()
    rospy.loginfo("Starting empty space estimation client...")
    cls.run()
