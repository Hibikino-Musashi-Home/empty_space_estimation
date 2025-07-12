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
import actionlib
from typing import Sequence
# import seg_node  # Assuming seg_node is a module that provides the main_segmentation function
from empty_space_estimation import seg_node 
from empty_space_estimation.msg import EmptySpace, EmptySpaceEstimationAction, EmptySpaceEstimationFeedback, EmptySpaceEstimationResult, EmptySpaceEstimationGoal
from tamlib.utils import Logger

import yaml

class SpaceEstimationClient(Logger):
    def __init__(self):
        self.bridge = CvBridge()
        Logger.__init__(self)

        rospy.wait_for_service("/empty_space_estimation/service")

        # ServiceProxy登録
        self.client = rospy.ServiceProxy("/empty_space_estimation/service", EmptySpaceService)
        self.client_nanosam = rospy.ServiceProxy("/object_detection_nanosam/service", ObjectDetectionService)
        rospy.loginfo("Empty Space Estimation Client initialized.")

        self.empty_space_estimation_client = actionlib.SimpleActionClient('/empty_space_estimation/action', EmptySpaceEstimationAction)
        rospy.loginfo("Action Server 待ち合わせ中…")
        self.empty_space_estimation_client.wait_for_server()
        rospy.loginfo("Action Server connected.")

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
     
        result_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
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
        input_path = config["PATH"]["IMG_TARGET"]
        try:
            if ros_image_msg.encoding == "8UC3":
                ros_image_msg.encoding = "bgr8"
                cv_image = self.bridge.imgmsg_to_cv2(ros_image_msg, desired_encoding='bgr8')
            else:
                cv_image = self.bridge.imgmsg_to_cv2(ros_image_msg, desired_encoding=ros_image_msg.encoding)

        except:
            rospy.loginfo("Encoding is not 8UC3, cannot convert to bgr8.")
            cv_image = self.bridge.compressed_imgmsg_to_cv2(ros_image_msg, desired_encoding='bgr8')

            

        # ROS Imageメッセージ → OpenCV形式に変換（BGR8）
       
        # JPEGで保存
        cv2.imwrite(input_path, cv_image)

        rospy.loginfo(f"Saving image to {input_path}")

        return input_path
    
    def get_result(self):
        rospy.loginfo("ESE-client: Waiting for empty space estimation result...")
        self.empty_space_estimation_client.wait_for_result()
        result = self.empty_space_estimation_client.get_result()
        rospy.loginfo("ESE-client: Get empty space estimation result")
        self.loginfo(result)
        return result.results

    def run(self, input_image_path=None, bbox=None, sync=False):
        try:
            # nanosamを呼び出す
            package = roslib.packages.get_pkg_dir("empty_space_estimation")
            ymal_path = os.path.join(package, "io", "nanosam.yaml")
            point = rospy.wait_for_message("/hma_pcl_reconst/depth_registered/points", PointCloud2, timeout=10)
            if input_image_path is None or bbox is None:  # input_image_pathが指定されていない場合
                request_nanosam = ObjectDetectionServiceRequest(use_latest_image=True, specific_id=ymal_path, max_distance=1.0)
                response_nanosam = self.client_nanosam(request_nanosam)
                bbox = response_nanosam.detections.bbox
                input_image_path = self.save_image(response_nanosam.detections.rgb)
            else: # input_image_pathが指定されている場合
                pass
            image_path = seg_node.main_segmentation(input_image_path)
            use_image = self.bridge.cv2_to_imgmsg(cv2.imread(image_path), encoding="bgr8")

            
            detection_image = self.mark_image(use_image, bbox)  # bboxを使用して画像にマークを付ける            

            # 空き領域推定サービス呼び出し
            if sync:
                request = EmptySpaceServiceRequest()
                # #  画像を保存
                # package_path = roslib.packages.get_pkg_dir("empty_space_estimation")
                # file_path = os.path.join(package_path, "input", "tmp_input_image.jpg")
                # self.save_image(self.image_msg, file_path)
                target_obj = rospy.get_param("/target_object", "コップ")
                add_prompt = rospy.get_param("/add_prompt", "")
                rospy.loginfo(f"Target object for space estimation: {target_obj}")
                request.question = f"追加情報として，{add_prompt}．棚の画像を解析して、{target_obj}置くのに適した場所を提案してください。"
                request.image = self.bridge.cv2_to_imgmsg(detection_image, encoding="bgr8")
                request.point = point  # PointCloud2メッセージをセット
                response = self.client(request)
                self.loginfo("空き領域推定結果:")
                self.loginfo(response.results)
                return response.results

            else:
                goal = EmptySpaceEstimationGoal()
                # #  画像を保存
                # package_path = roslib.packages.get_pkg_dir("empty_space_estimation")
                # file_path = os.path.join(package_path, "input", "tmp_input_image.jpg")
                # self.save_image(self.image_msg, file_path)
                target_obj = rospy.get_param("/target_object", "コップ")
                add_prompt = rospy.get_param("/add_prompt", "")
                self.loginfo(f"Target object for space estimation: {target_obj}")
                goal.question = f"追加情報として，{add_prompt}．棚の画像を解析して、{target_obj}置くのに適した場所を提案してください。"
                goal.image = self.bridge.cv2_to_imgmsg(detection_image, encoding="bgr8")
                goal.point = point  # PointCloud2メッセージをセット
                self.loginfo("Sending goal to empty space estimation action server...")
                self.empty_space_estimation_client.send_goal(goal)
                self.empty_space_estimation_client.wait_for_result()
                result = self.empty_space_estimation_client.get_result()
                print(result)
                self.loginfo("Sended goal to empty space estimation action server...")

        except rospy.ServiceException as e:
            rospy.logerr("サービス呼び出しに失敗しました: %s", str(e))


if __name__ == "__main__":
    rospy.init_node("empty_space_client")
    cls = SpaceEstimationClient()
    rospy.loginfo("Starting empty space estimation client...")
    # image_path = "/home/hma/ros_ws/src/7_tasks/hma_hsr_sg_pkg/io/images/20250709_1502/place_0.png"
    # cls.run(input_image_path=image_path)  # Noneを渡すことで最新の画像を使用
    cls.run(sync=False)

    cls.get_result()
