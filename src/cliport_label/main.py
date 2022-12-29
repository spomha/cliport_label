"""Main program logic for tool"""
import rospy

from cliport_label.camera import CameraStream
from cliport_label.taskexecutor import TaskExecutor
from cliport_label.gui import GUI


def main(config) -> None:
    """Main program flow"""
    # Create ros node
    rospy.init_node("cliport_label", anonymous=True, log_level=rospy.INFO)
    # Initialize classes
    streamer = CameraStream()
    taskexecutor = None
    if config["taskexecutor"]["enable"]: 
        taskexecutor = TaskExecutor(config)
    gui = GUI(config, streamer, taskexecutor)
    # Main program loop
    try:
        while not rospy.is_shutdown():
            gui.run()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down cliport label tool")
    gui.cleanup()
