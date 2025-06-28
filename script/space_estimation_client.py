#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, PointCloud2
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
import seg_node  # Assuming seg_node is a module that provides the main_segmentation function
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

    def save_image(self, image_msg, file_path):
        try:
            image_msg.encoding = "bgr8"  # 画像のエンコーディングを指定

            # ROSのImageメッセージをOpenCVの画像に変換
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            # 画像を保存
            cv2.imwrite(file_path, cv_image)
            rospy.loginfo(f"画像を保存しました: {file_path}")
        except Exception as e:
            rospy.logerr(f"画像の保存に失敗しました: {str(e)}")

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
     
        result_image = self.bridge.compressed_imgmsg_to_cv2(image)
        for box in bboxes:  # msg.bbox: List of this custom message
            x_min = box.x - box.w / 2
            y_min = box.y - box.h / 2
            x_max = box.x + box.w / 2
            y_max = box.y + box.h / 2
            bbox = torch.tensor([x_min, y_min, x_max, y_max]).round().int().tolist()
            cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        return result_image
    
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

        return 
    

    def run(self):
        try:
            # nanosamを呼び出す
            package = roslib.packages.get_pkg_dir("empty_space_estimation")
            ymal_path = os.path.join(package, "io", "nanosam.yaml")
            rospy.loginfo("Nanosam detection service is ready!")
            point = rospy.wait_for_message("/hma_pcl_reconst/depth_registered/points", PointCloud2, timeout=10)
            request_nanosam = ObjectDetectionServiceRequest(use_latest_image=True, specific_id=ymal_path, max_distance=1.0)
            response_nanosam = self.client_nanosam(request_nanosam)
            input_image_path = self.save_image(response_nanosam.detections.rgb)
            image_path = seg_node.main_segmentation(input_image_path)
            use_image = self.bridge.cv2_to_imgmsg(cv2.imread(image_path), encoding="bgr8")

            
            detection_image = self.mark_image(use_image, response_nanosam.detections.bbox)
            
            

            # 空き領域推定サービス呼び出し
            request = EmptySpaceServiceRequest()
            # #  画像を保存
            # package_path = roslib.packages.get_pkg_dir("empty_space_estimation")
            # file_path = os.path.join(package_path, "input", "tmp_input_image.jpg")
            # self.save_image(self.image_msg, file_path)
            
            request.image = self.bridge.cv2_to_imgmsg(detection_image, encoding="bgr8")
            request.point = point  # PointCloud2メッセージをセット
            response = self.client(request)

            rospy.loginfo("空き領域推定結果:")
            rospy.loginfo(response.results)

        except rospy.ServiceException as e:
            rospy.logerr("サービス呼び出しに失敗しました: %s", str(e))


if __name__ == "__main__":
    cls = SpaceEstimationClient()
    rospy.loginfo("Starting empty space estimation client...")
    cls.run()
