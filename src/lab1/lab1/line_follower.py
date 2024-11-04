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
        super().__init__("line_follower")

        # Init utils node (LED + LCD)
        self.utils = UTILS(self)

        # Init recieving image from rae
        self.br = CvBridge()
        self.image_subscriber = self.create_subscription(
            Image, "/rae/right/image_raw", self.image_callback, 10
        )
        self.image_subscriber
        self.image = None

        self.timer = self.create_timer(
            timer_period_sec=0.06, callback=self.timer_callback
        )
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
        return (0, height, 0, width)

    def get_lines(self, mask, out):
        """
        Detect lines using hough transform
        Mask: Binary mask to detect lines on
        Out: Original images to draw line on

        Return list of lines, each line is represented by a 4-element vector (x1, y1, x2, y2)
        """
        # CV2 Hough transform
        linesP = cv2.HoughLinesP(
            mask,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=90,
            maxLineGap=5,
        )

        # Draw lines on original image
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(out, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 4)
        else:
            linesP = []

        return linesP

    def image_callback(self, data):
        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(data)
        # TODO Image corretion code here
        self.image = current_frame

    def edge_detector(self, current_frame):
        # TODO Maybe import this from edge_detector.py with a command line argument to do different methods

        # Convert the image to grayscale
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to binarize the image (only keep bright areas)
        _, binary = cv2.threshold(current_frame, 250, 255, cv2.THRESH_BINARY)

        # Setting parameter values
        t_lower = 50  # Lower Threshold
        t_upper = 150  # Upper threshold

        # Applying the Canny Edge filter
        canny = cv2.Canny(binary, t_lower, t_upper)

        # Dilate the edges to make it easier for Hough transform
        kernel = np.ones((7, 7), np.uint8)
        edges_dilated = cv2.dilate(canny, kernel)
        return edges_dilated, binary

    def timer_callback(self):
        # Wait for the first image to be received
        if type(self.image) != np.ndarray:
            return

        # Get image stats
        image = self.image.copy()
        self.height = image.shape[0]
        self.width = image.shape[1]
        # Edge detection
        mask, binary = self.edge_detector(image)
        image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        # Crop only top part to detect lines from
        crop_h_start, crop_h_stop, crop_w_start, crop_w_stop = self.crop_size(
            self.height, self.width
        )
        crop = mask[crop_h_start:crop_h_stop, crop_w_start:crop_w_stop]

        # Get lines using hough and draw on image crop
        lines = self.get_lines(
            crop, image[crop_h_start:crop_h_stop, crop_w_start:crop_w_stop]
        )

        # Get closest line
        error = np.inf
        min_i = None
        for i in range(0, len(lines)):
            line = lines[i][0]
            line_center_w = crop_w_start + ((line[0] + line[2]) / 2)
            line_error = line_center_w - (self.width / 2)

            if line_error < error:
                error = line_error
                min_i = i
        if min_i:
            line = lines[min_i][0]
            cv2.circle(
                image,
                (int(crop_w_start + ((line[0] + line[2]) / 2)), 0),
                5,
                (0, 255, 0),
                7,
            )
            # Draw middle point of closest line
            cv2.circle(
                image,
                (
                    int(crop_w_start + (line[0] + line[2]) / 2),
                    int(crop_h_start + (line[1] + line[3]) / 2),
                ),
                5,
                (0, 255, 0),
                7,
            )

        # Draw box which has been cropped
        cv2.rectangle(
            image,
            (crop_w_start, crop_h_start),
            (crop_w_stop, crop_h_stop),
            (255, 0, 0),
            2,
        )

        # Draw number of lines detected to RAE screen
        self.utils.draw_text(f"{len(lines)} lines")

        # Show the output image to the user
        cv2.imshow("output", image)
        # Print the image for 5milis, then resume execution
        cv2.waitKey(5)

        # TODO Turn robot towards center of closest line and drive forward
        # TODO Define stop condition


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


if __name__ == "__main__":
    main()
