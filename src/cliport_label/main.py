"""Main program logic for tool"""
import pickle
import rospy

from cliport_label.camera import CameraStream
from cliport_label.taskexecutor import TaskExecutor
from cliport_label.gui import ToolGUI, ViewerGUI


def main_tool(config) -> None:
    """Main tool program flow"""
    # Create ros node
    rospy.init_node("cliport_label", anonymous=True, log_level=rospy.INFO)
    # Initialize classes
    streamer = CameraStream()
    taskexecutor = None
    if config["taskexecutor"]["enable"]: 
        taskexecutor = TaskExecutor(config)
    gui = ToolGUI(config, streamer, taskexecutor)
    # Main program loop
    try:
        while not rospy.is_shutdown():
            gui.run()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down cliport label tool")
    gui.cleanup()


def main_viewer(config) -> None:
    """Main viewer program flow"""
    gui = ViewerGUI(config)
    # Main program loop
    try:
        while not rospy.is_shutdown():
            gui.run()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down cliport label viewer")
    gui.cleanup()


def main_editor(filepath, lang_goal) -> None:
    """Main editor program flow"""
    if lang_goal != "":
        with open(filepath, "rb") as fd:
            data = pickle.load(fd)
            old_lang_goal = data['info']['lang_goal']
            print(f"Changing {old_lang_goal =} to {lang_goal = }")
            data['info']['lang_goal'] = lang_goal
        
        with open(filepath, "wb") as fd:
            pickle.dump(data, fd)
