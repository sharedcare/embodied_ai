import os
from typing import List, Optional, Tuple
import numpy as np
import cv2

from PIL import Image
from autogen import ConversableAgent


DEFAULT_ROBOT_SYS_MSG = "You are a robot agent."


class RobotAgent(ConversableAgent):
    def __init__(
        self,
        name: str,
        system_message: Optional[Tuple[str, List]] = DEFAULT_ROBOT_SYS_MSG,
        *args,
        **kwargs,
    ):
        """
        Args:
            name (str): agent name.
            system_message (str): system message for the ChatCompletion inference.
                Please override this attribute if you want to reprogram the agent.
            **kwargs (dict): Please refer to other kwargs in
                [ConversableAgent](../conversable_agent#__init__).
        """
        super().__init__(
            name,
            system_message=system_message,
            *args,
            **kwargs,
        )

    """Robot Proxy Agent"""

    def get_image(self):
        """Get RGB image data from the robot's camera"""
        if os.path.exists(".tmp/image.png"):
            return cv2.imread(".tmp/image.png")
        else:
            cam = cv2.VideoCapture(0)
            result, image = cam.read()
            return image

    def get_object_position(self, bounding_box):
        """Get object position in the real world with (x, y, z)"""
        x0 = bounding_box[0]
        y0 = bounding_box[1]
        x1 = bounding_box[2]
        y1 = bounding_box[3]
        x = (x0 + x1) / 2
        y = (y0 + y1) / 2
        z = 10
        object_pos = (x, y, 10)
        return object_pos

    def get_grasp_pose(self, bounding_box):
        """Get grasp pose for the robot arm with [x, y, z, roll, pitch, yaw, gripper_closed]"""
        x0 = bounding_box[0]
        y0 = bounding_box[1]
        x1 = bounding_box[2]
        y1 = bounding_box[3]
        x = (x0 + x1) / 20
        y = (y0 + y1) / 20
        z = 1
        roll, pitch, yaw = [0, np.pi / 2, np.pi]
        grasp_pose = [x, y, z, roll, pitch, yaw, 1]
        return grasp_pose

    def robot_execute(self, object_name, grasp_pose, place_position):
        print(f"Place {object_name} on {place_position}")
        ...
        status = "success"
        return status
