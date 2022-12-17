"""Camera handling"""
# Ros libraries
import rospy
from cv_bridge import CvBridge

# Ros Messages
from sensor_msgs.msg import Image


class CameraStream:

    def __init__(self):
        """Initialize ros subscriber"""

        # subscribed Topic
        topic_rgb = "/camera/color/image_raw"
        topic_depth = "/camera/aligned_depth_to_color/image_raw"
        self.subscriber_rgb = rospy.Subscriber(topic_rgb,
            Image, self.callback_rgb, queue_size=1, buff_size=2**32)
        self.subscriber_depth = rospy.Subscriber(topic_depth,
            Image, self.callback_depth, queue_size=1, buff_size=2**32)

        self.bridge = CvBridge()
        self.rgb = self.depth = None
        rospy.loginfo(f"Subscribed to {topic_rgb}")
        rospy.loginfo(f"Subscribed to {topic_depth}")


    def callback_rgb(self, data):
        """RGB callback"""
        self.rgb = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")

    def callback_depth(self, data):
        """Depth callback"""
        self.depth = self.bridge.imgmsg_to_cv2(data,)
