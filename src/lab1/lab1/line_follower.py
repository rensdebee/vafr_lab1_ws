import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

import numpy as np
import cv2
from cv_bridge import CvBridge
from lab1.utils import UTILS
import signal


class LineFollower(Node):
    def __init__(self):
        # Set instructions
        self.should_move = False
        self.display_gray = False
        self.turn_speed = 0.2
        self.forward_speed = 0.2

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

        # Set leds
        self.utils.set_leds("#f2c40c", brightness=10)
        self.utils.draw_text("Setting up..")
        self.frame = 0

        # Running line following on fixed timing
        self.timer = self.create_timer(
            timer_period_sec=0.06, callback=self.timer_callback
        )

        # Publish movement commands
        self.publisher = self.create_publisher(
            Twist, "/cmd_vel", rclpy.qos.qos_profile_system_default
        )

        # On keyboard interrupt
        signal.signal(signal.SIGINT, self.stop)

    def crop_size(self, image):
        """
        Get the measures to crop the image
        Output:
        (Height_upper_boundary, Height_lower_boundary,
        Width_left_boundary, Width_right_boundary)
        """
        height = image.shape[0]
        width = image.shape[1]

        # Step 1: Define the points for the triangle (adjust coordinates as needed)
        triangle_up_left = (0, 0)
        triangle_up_right = (width, 0)
        triangle_down = (width // 2, height)
        self.triangle = np.array(
            [[triangle_up_left, triangle_up_right, triangle_down]],
            dtype=np.int32,
        )

        # Step 2: Create a mask with the same size as the image, initially filled with 0 (black)
        mask = np.zeros_like(image)

        # Fill the triangular area on the mask with white (255)
        cv2.fillPoly(mask, self.triangle, 255)

        # Step 3: Apply the mask to get the triangular ROI
        triangular_roi = cv2.bitwise_and(image, mask)

        # Define point to focus on for error
        self.w_focus = triangle_up_right[0] - triangle_up_left[0]
        self.h_focus = triangle_up_right[1] - triangle_up_left[1] - 100

        return triangular_roi

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
            threshold=200,
            minLineLength=80,
            maxLineGap=150,
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
        # TODO Read from file
        mtx = np.array(
            [
                [311.717726, 0.000000, 307.396186],
                [0.000000, 313.319557, 191.237882],
                [0.000000, 0.000000, 1.000000],
            ]
        )
        dist = np.array([-0.244207, 0.043980, 0.000955, -0.000878, 0.000000])
        h = current_frame.shape[0]
        w = current_frame.shape[1]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
        self.image = cv2.undistort(current_frame, mtx, dist, None, newcameramtx)

    def edge_detector(self, current_frame):
        # Convert the image to grayscale
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to binarize the image (only keep bright areas)
        _, binary = cv2.threshold(current_frame, 200, 255, cv2.THRESH_BINARY)

        # TODO Maybe import this from edge_detector.py with a command line argument to do different methods
        # Setting parameter values
        t_lower = 50  # Lower Threshold
        t_upper = 150  # Upper threshold
        # Applying the Canny Edge filter
        canny = cv2.Canny(binary, t_lower, t_upper)

        # Dilate the edges to make it easier for Hough transform
        kernel = np.ones((10, 10), np.uint8)
        edges_dilated = cv2.dilate(canny, kernel)
        return edges_dilated, binary

    def get_closest_line(
        self,
        lines,
    ):
        # Get closest line
        abs_error = np.inf
        rot_error = None
        min_i = None

        for i in range(0, len(lines)):
            line = lines[i][0]
            line_center_w = (line[0] + line[2]) / 2
            line_center_h = (line[1] + line[3]) / 2

            line_error_w = self.w_focus - line_center_w
            line_error_h = self.h_focus - line_center_h

            abs_total_error = np.abs(line_error_w) + 2 * np.abs(line_error_h)

            if abs_total_error < abs_error:
                abs_error = abs_total_error
                rot_error = line_error_w
                min_i = i

        # Return line with lowest error and width error
        if min_i is not None:
            return lines[min_i][0], rot_error
        else:
            return None, None

    def timer_callback(self):
        # Wait for the first image to be received
        if type(self.image) != np.ndarray:
            return

        # Get image stats
        image = self.image.copy()
        # cv2.imwrite(f"./imgs/frame_{self.frame}.png", image)
        self.frame += 1
        # Edge detection
        edges_dilated, binary_image = self.edge_detector(image)

        # Display binary image
        if self.display_gray:
            image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

        # Crop only top part to detect lines from
        triangular_roi = self.crop_size(edges_dilated.copy())

        # Get lines using hough and draw on image crop
        lines = self.get_lines(triangular_roi, image)
        line, rot_error = self.get_closest_line(lines)

        if line is not None:
            self.utils.set_leds("#00FF00")
            # < 0  IS RIGHT > 0 IS LEFT
            # TODO base turning on error with posible clip
            turn = rot_error

            if turn < 0:
                self.utils.draw_text("Going Right")
                print("Right")
                turn = -1 * self.turn_speed
            else:
                self.utils.draw_text("Going Left")
                turn = self.turn_speed
                print("Left")

            if self.should_move:
                message = Twist()
                message.angular.z = turn
                message.linear.x = self.forward_speed
                self.publisher.publish(message)
            else:
                self.publisher.publish(Twist())

            # Draw middle point of closest line
            cv2.circle(
                image,
                (
                    int((line[0] + line[2]) / 2),
                    int((line[1] + line[3]) / 2),
                ),
                10,
                (0, 255, 0),
                7,
            )
        else:
            self.utils.set_leds("#FF0000")
            # Draw number of lines detected to RAE screen
            self.utils.draw_text(f"{len(lines)} lines")
            if self.should_move:
                message = Twist()
                message.angular.z = self.turn_speed
                self.publisher.publish(message)
            else:
                self.publisher.publish(Twist())

        # Draw box which has been cropped
        cv2.polylines(
            image, self.triangle, isClosed=True, color=(255, 0, 0), thickness=2
        )

        # Draw point used for error calculation
        cv2.circle(
            image,
            (int(self.w_focus), int(self.h_focus)),
            5,
            (0, 255, 255),
            7,
        )

        # Show the output image to the user
        cv2.imshow("output", image)
        # Print the image for 5milis, then resume execution
        cv2.waitKey(5)

        # TODO Define stop condition

    def stop(self, signum=None, frame=None):
        self.utils.set_leds("#ce10e3")
        self.publisher.publish(Twist())
        self.utils.draw_text(f"Shut down")
        self.destroy_node()
        rclpy.shutdown()
        exit()


def main(args=None):

    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    line_follower = LineFollower()

    try:
        # Spin the node to call callback functions
        rclpy.spin(line_follower)
    except KeyboardInterrupt:
        line_follower.get_logger().info("Keyboard interrupt caught in main loop.")
    finally:
        # Ensure node is properly destroyed and stopped on shutdown
        line_follower.destroy_node()
        line_follower.stop()


if __name__ == "__main__":
    main()
