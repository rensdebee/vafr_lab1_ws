# Basic ROS 2 program to subscribe to real-time streaming 
# video from your built-in webcam
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com
  
# Import the necessary libraries
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # Image is the message type
from std_msgs.msg import ColorRGBA
from rae_msgs.msg import LEDControl
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from time import sleep
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library

from lab1.utils import UTILS
 
class ImageSubscriber(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('image_subscriber')
      
    # Create the subscriber. This subscriber will receive an Image
    # from the video_frames topic. The queue size is 10 messages.
    self.subscription = self.create_subscription(
      Image, 
      '/rae/right/image_raw', 
      self.listener_callback, 
      40)
    self.subscription # prevent unused variable warning
    
    self.publisher = self.create_publisher(Twist, '/cmd_vel', rclpy.qos.qos_profile_system_default)

    self.utils = UTILS(self)
    self.frame_num = 0
    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()
    self.LINEAR_SPEED = 0.05
  
  def listener_callback(self, data):
    """
    Callback function.
    """
    # Display the message on the console
    self.frame_num += 1
    self.get_logger().info(f'speed: {self.LINEAR_SPEED}')
 
    # Convert ROS Image message to OpenCV image
    current_frame = self.br.imgmsg_to_cv2(data)

    # Convert the image to grayscale
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to binarize the image (only keep bright areas)
    _, current_frame = cv2.threshold(current_frame, 200, 255, cv2.THRESH_BINARY)

    # Setting parameter values 
    t_lower = 50  # Lower Threshold 
    t_upper = 150  # Upper threshold 
    
    # Applying the Canny Edge filter 
    current_frame = cv2.Canny(current_frame, t_lower, t_upper) 

    # Updating robot
    self.utils.draw_text(f"frame: {self.frame_num}")

    # Display image
    cv2.imshow("camera", current_frame)
    cv2.waitKey(1)
    if self.frame_num % 30 == 0:
      self.utils.set_leds(self.utils.random_hex_color())
      self.LINEAR_SPEED = self.LINEAR_SPEED * -1
    message = Twist()
    message.linear.x = self.LINEAR_SPEED
    self.publisher.publish(message)

  
def main(args=None):
  
  # Initialize the rclpy library
  rclpy.init(args=args)

  # Create the node
  image_subscriber = ImageSubscriber()
  
  # Spin the node so the callback function is called.
  rclpy.spin(image_subscriber)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  image_subscriber.destroy_node()
  
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()
