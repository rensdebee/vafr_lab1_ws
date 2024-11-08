import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

import numpy as np
import cv2
from cv_bridge import CvBridge
from lab1.utils import UTILS
from lab1.calibrate import undistort_from_saved_data
import signal
import math


class LineFollower(Node):
    def __init__(self):
        # Send move instructions
        self.should_move = False

        # Choose which image to use as an underground
        self.display_gray = False
        self.display_canny = False
        self.display_dilated_canny = False
        self.display_roi = False

        # Save each 60 frames
        self.save_frames = False
        self.save_raw = True

        # Diferent pipeline options
        self.undistort = False
        self.binarize = True
        self.dilate = True
        self.roi = True
        self.choose_line = True

        # Movement speeds
        self.turn_speed = 0.5
        self.forward_speed = 0.2

        # Init node
        super().__init__("line_follower")

        # Init utils node (LED + LCD)
        self.utils = UTILS(self)

        # Init recieving image from rae
        self.br = CvBridge()
        # self.image_subscriber = self.create_subscription(
        #     Image, "/rae/right/image_raw", self.image_callback, 10
        # )
        self.image_subscriber = self.create_subscription(
            CompressedImage, "/rae/right/image_raw/compressed", self.image_callback, 10
        )
        self.image_subscriber
        self.image = None

        # Set leds
        self.utils.set_leds("#f2c40c")
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

    def image_callback(self, data):
        # Convert ROS Image message to OpenCV image
        current_frame = self.br.compressed_imgmsg_to_cv2(data)
        # Undistort image using found calibration
        self.image = current_frame
        if self.undistort:
            self.image = undistort_from_saved_data(
                "./src/lab1/lab1/calibration_data.npz", current_frame
            )

    def crop_size(self, image):
        """
        Get the measures to crop the image
        Output:
        (Height_upper_boundary, Height_lower_boundary,
        Width_left_boundary, Width_right_boundary)
        """
        height = image.shape[0]
        width = image.shape[1]

        # Step 1: Define the points for the triangle
        # order must be kept otherwise downstream will break
        triangle_up_left = (0 + 100, 0)
        triangle_up_right = (width - 100, 0)
        triangle_down = (width // 2, height - 80)
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
        self.w_focus = (triangle_up_right[0] + triangle_up_left[0]) // 2
        self.h_focus = ((triangle_up_right[1] + triangle_up_left[1]) // 2) + 100
        if not self.roi:
            triangular_roi = image

        return triangular_roi

    def edge_detector(self, current_frame):
        # Convert the image to grayscale
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to binarize the image (only keep bright areas)
        binary = current_frame
        if self.binarize:
            _, binary = cv2.threshold(current_frame, 240, 255, cv2.THRESH_BINARY)
            binary = cv2.bitwise_and(current_frame, binary)

        # TODO Maybe import this from edge_detector.py with a command line argument to do different methods
        # Setting parameter values
        t_lower = 50  # Lower Threshold
        t_upper = 150  # Upper threshold
        # Applying the Canny Edge filter
        canny = cv2.Canny(binary, t_lower, t_upper)

        # Dilate the edges to help for hough transform
        edges_dilated = canny
        if self.dilate:
            kernel = np.ones((15, 15), np.uint8)
            edges_dilated = cv2.dilate(canny, kernel)

        return edges_dilated, binary, canny

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
            threshold=175,
            minLineLength=100,
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

    def point_distance(self, p1, p2):
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def distance_point_to_line(self, line_start, line_end, focus_point):
        x1, y1 = line_start
        x2, y2 = line_end
        x0, y0 = focus_point
        # Calculate the direction vector AB
        dx, dy = x2 - x1, y2 - y1

        # If A and B are the same point, we can't define a line, so return the distance PA
        if dx == 0 and dy == 0:
            return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

        # Calculate the projection scalar t for the point P on line AB
        t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx**2 + dy**2)

        # Find the coordinates of the projection point Q on the line
        qx, qy = x1 + t * dx, y1 + t * dy

        # Calculate the distance between P and Q
        distance = math.sqrt((qx - x0) ** 2 + (qy - y0) ** 2)
        return distance

    def get_closest_line(
        self,
        lines,
    ):
        # Get closest line
        abs_error = np.inf
        rot_error = None
        min_i = None

        # Calculate line which center is clossst to our focus point
        for i in range(0, len(lines)):
            line = lines[i][0]
            line_center_w = (line[0] + line[2]) / 2
            line_center_h = (line[1] + line[3]) / 2

            # Width and Height error
            line_error_w = self.w_focus - line_center_w
            line_error_h = self.h_focus - line_center_h

            # Penalize height error more
            # abs_total_error = np.abs(line_error_w) + 2 * np.abs(line_error_h)
            abs_total_error = self.distance_point_to_line(
                (line[0], line[1]), (line[2], line[3]), (self.w_focus, self.h_focus)
            )
            # Get line with lowest error
            if abs_total_error < abs_error:
                abs_error = abs_total_error
                rot_error = line_error_w
                min_i = i

        # Return line with lowest error and width error
        if min_i is not None:
            # Get line with lowest error
            line = lines[min_i][0]
            # Get start and end point of line
            p_1 = (line[0], line[1])
            p_2 = (line[2], line[3])

            # Lower part of triangle
            distant_point = (self.triangle[0][2][0], self.triangle[0][2][1])

            # Distance between start, end point of line and lower part of triangle
            dist_1 = self.point_distance(distant_point, p_1)
            dist_2 = self.point_distance(distant_point, p_2)

            # Set followpoint to end point of line which is closest to lower part of triangle
            if dist_1 < dist_2:
                follow_point = p_1
            else:
                follow_point = p_2

            # Calculate direction to turn based on difference between width error of focus point and point to follow
            rot_error = self.w_focus - follow_point[0]
            return lines[min_i][0], rot_error, follow_point
        else:
            return None, None, None

    def timer_callback(self):
        # Wait for the first image to be received
        if type(self.image) != np.ndarray:
            return

        # Get image stats
        image = self.image.copy()
        raw_image = self.image.copy()
        # cv2.imwrite(f"./imgs/frame_{self.frame}.png", image)
        self.frame += 1
        # Edge detection
        edges_dilated, binary_image, canny = self.edge_detector(image)

        # Choose image to display
        if self.display_gray:
            image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        elif self.display_canny:
            image = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        elif self.display_dilated_canny:
            image = cv2.cvtColor(edges_dilated, cv2.COLOR_GRAY2BGR)

        # Crop only triangular part to detect lines from
        triangular_roi = self.crop_size(edges_dilated.copy())
        if self.display_roi:
            image = cv2.cvtColor(triangular_roi, cv2.COLOR_GRAY2BGR)

        # Get lines using hough and draw on image crop
        lines = self.get_lines(triangular_roi, image)

        # Draw number of lines detected to RAE screen
        self.utils.draw_text(f"{len(lines)} lines")
        if self.get_closest_line:
            line, rot_error, follow_point = self.get_closest_line(lines)

        if line is not None and self.choose_line:
            # < 0  IS RIGHT > 0 IS LEFT
            turn = rot_error

            if turn < 0:
                self.utils.set_leds("#00FF00")
                print("Right")
                turn = -1 * self.turn_speed
            else:
                self.utils.set_leds("#FF0000")
                turn = 1 * self.turn_speed
                print("Left")

            if self.should_move:
                message = Twist()
                message.angular.z = turn
                message.linear.x = self.forward_speed
                self.publisher.publish(message)
            else:
                self.publisher.publish(Twist())

            # Draw closest line in green
            cv2.line(
                image,
                (line[0], line[1]),
                (line[2], line[3]),
                (0, 255, 0),
                5,
            )
            # Draw Follow point
            cv2.circle(
                image,
                (
                    int(follow_point[0]),
                    int(follow_point[1]),
                ),
                10,
                (255, 255, 0),
                7,
            )
        else:
            self.utils.set_leds("#f2c40c")
            if self.should_move:
                message = Twist()
                message.angular.z = self.turn_speed * 5
                self.publisher.publish(message)
            else:
                self.publisher.publish(Twist())

        # Draw box which has been cropped
        if self.roi:
            cv2.polylines(
                image, self.triangle, isClosed=True, color=(255, 0, 0), thickness=2
            )

        # Draw point used for error calculation
        if self.choose_line:
            cv2.circle(
                image,
                (int(self.w_focus), int(self.h_focus)),
                5,
                (0, 255, 255),
                7,
            )
        if self.frame % 60 == 1 and (self.save_frames or self.save_raw):
            print(f"Saving frame {self.frame}")
            if self.save_raw:
                cv2.imwrite(f"./frame_{raw_image}.png", image)
            else:
                cv2.imwrite(f"./frame_{self.frame}.png", image)
        self.frame += 1

        # Show the output image to the user
        cv2.imshow("output", image)
        # Print the image for 5milis, then resume execution
        cv2.waitKey(5)

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
