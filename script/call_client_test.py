#!/usr/bin/env python3
import rospy
from empty_space_estimation.space_estimation_client import SpaceEstimationClient

def your_task_function():
    # 1. 配置対象を設定
    rospy.set_param("/target_obj", "コップ")

    # 2. クライアント作成
    client = SpaceEstimationClient()

    # 3. 推定実行
    results = client.run()
    rospy.loginfo(results)
    

if __name__ == "__main__":
    your_task_function()
    rospy.spin()