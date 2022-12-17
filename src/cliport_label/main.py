"""Main program logic for tool"""
import rospy

from cliport_label.camera import CameraStream
from cliport_label.gui import GUI


def main() -> None:
    """Main program flow"""
    # Create ros node
    rospy.init_node("cliport_label", anonymous=True, log_level=rospy.INFO)
    # Initialize classes
    stream = CameraStream()
    gui = GUI()
    # Main program loop
    try:
        while not rospy.is_shutdown():
            gui.run(stream)
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down cliport label tool")
    gui.cleanup()
