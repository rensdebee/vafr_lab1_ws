import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

import numpy as np
import cv2
from cv_bridge import CvBridge
from lab1.utils import UTILS

class LineFollower(Node):
    def __init__(self):
        # Init node
        super().__init__('line_follower')

        # Init utils node (LED + LCD)
        self.utils = UTILS(self)
        
        # Init recieving image from rae
        self.br = CvBridge()
        self.image_subscriber = self.create_subscription(Image, "/rae/right/image_raw", self.image_callback, 10)
        self.image_subscriber
        self.image = None

        self.timer = self.create_timer(timer_period_sec=0.06, callback=self.timer_callback)
        self.should_move = False
        self.finalization_countdown = None

        self.utils.set_leds("#f2c40c", brightness=10)
        self.utils.draw_text("Setting up..")

    def crop_size(self, height, width):
        """
        Get the measures to crop the image
        Output:
        (Height_upper_boundary, Height_lower_boundary,
        Width_left_boundary, Width_right_boundary)
        """
        ## Update these values to your liking.

        return (0, height//2, width//4, 3*width//4)
    
    def get_contour_data(self, mask, out):
        linesP = cv2.HoughLinesP(mask, rho=1, theta=np.pi / 180, threshold=250, minLineLength=90, maxLineGap=10)
    
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(out, (l[0], l[1]), (l[2], l[3]), (0,0,255), 4)
        else:
            linesP = []


        return linesP
    

    def image_callback(self, data):
        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(data)
        self.image = current_frame
    
    def edge_detector(self, current_frame):
        # Convert the image to grayscale
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to binarize the image (only keep bright areas)
        _, current_frame = cv2.threshold(current_frame, 150, 255, cv2.THRESH_BINARY)

        # Setting parameter values 
        t_lower = 50  # Lower Threshold 
        t_upper = 150  # Upper threshold 
        
        # Applying the Canny Edge filter 
        current_frame = cv2.Canny(current_frame, t_lower, t_upper) 
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(current_frame, kernel)
        return edges_dilated
    
    def timer_callback(self):
        # Wait for the first image to be received
        if type(self.image) != np.ndarray:
            return
        
        image = self.image.copy()
        self.height = image.shape[0] 
        self.width = image.shape[1]

        crop_h_start, crop_h_stop, crop_w_start, crop_w_stop = self.crop_size(self.height, self.width)
        mask = self.edge_detector(image)
        # get the bottom part of the image (matrix slicing)
        crop = mask[crop_h_start:crop_h_stop, crop_w_start:crop_w_stop]


        # output = image
        lines = self.get_contour_data(crop, image[crop_h_start:crop_h_stop, crop_w_start:crop_w_stop])
        self.utils.draw_text(f"{len(lines)} lines")
        # Show the output image to the user
        cv2.imshow("output", image)
        # Print the image for 5milis, then resume execution
        cv2.waitKey(5)
        


def main(args=None):
  
  # Initialize the rclpy library
  rclpy.init(args=args)

  # Create the node
  line_follower = LineFollower()
  
  # Spin the node so the callback function is called.
  rclpy.spin(line_follower)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  line_follower.destroy_node()
  
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()

