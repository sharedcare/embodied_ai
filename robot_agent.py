import os
import numpy as np
import pyrealsense2 as rs
import cv2

from PIL import Image
from autogen import Agent


class RobotAgent(Agent):
    def __init__(self):
        super.__init__(self)
        # Create a pipeline
        self.pipeline = rs.pipeline()
    
        # Create a config object to configure the pipeline
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
    """Robot Proxy Agent"""
    def get_image():
        """Get RGB image data from the robot's camera
        """
        if os.path.exists('.tmp/color.png'):
            return Image.open('.tmp/color.png')
        
        return None

    def get_depth():
        """Get depth data from the robot's camera
        """
        if os.path.exists('.tmp/depth.npy'):
            return np.load('.tmp/depth.npy')
        
        return None

    def start_pipeline():
        # Start the pipeline
        self.pipeline.start(self.config)
        align = rs.align(rs.stream.color)  # Create align object for depth-color alignment

        num = 0
        try:
            while True:
                # Wait for a coherent pair of frames: color and depth
                frames = self.pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                if not aligned_frames:
                    continue  # If alignment fails, go back to the beginning of the loop
    
                color_frame = aligned_frames.get_color_frame()
                aligned_depth_frame = aligned_frames.get_depth_frame()
    
                if not color_frame or not aligned_depth_frame:
                    continue
    
                # Convert aligned_depth_frame and color_frame to numpy arrays
                aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
    
                # Display the aligned depth image
                aligned_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha=0.03),
                                                           cv2.COLORMAP_JET)

                np.save('.tmp/depth.npy', aligned_depth_image)
                cv2.imwrite('.tmp/color.png', color_image)

                num += 1

        finally:
            # Stop the pipeline and close all windows
            self.pipeline.stop()
            
    def stop_pipeline():
        self.pipeline.stop()
        
    def grasp(bbox, color=None, depth=None):
        pass
