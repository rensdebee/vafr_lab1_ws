from line_follower import LineFollower
import cv2
import rclpy

line_follower = LineFollower()

img = cv2.imread("./RAW/frame_61.png")

line_follower.image = img

line_follower.timer_callback()