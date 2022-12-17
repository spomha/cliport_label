"""GUI for label tool"""
from enum import Enum

import cv2
import numpy as np
import rospy

from cliport_label.taskexecutor import TaskInfo, TaskExecutor
from cliport_label.camera import CameraStream
from cliport_label.utils import get_line_theta, draw_on_disp_img

class StreamType(Enum):
    RGB = 0
    DEPTH = 1

class ToolType(Enum):
    PICKBBOX = 0
    PICKROTATE = 1
    PLACEBBOX = 2
    PLACEROTATE = 3
    NONE = 4

class GUI:
    """Visualize and interact with camera stream and snapshot window"""

    def __init__(self) -> None:
        """Initialize stuff"""
        self.stream_win = "stream"
        self.snapshot_win = "snapped_observation"
        self.stream_type = StreamType.RGB
        self.snapshot = None
        self.pick_data = {'rotation': 0, 'bbox': [], 'rotline': []}
        self.place_data = {'rotation': 0, 'bbox': [], 'rotline': []}
        self.selected_tool = ToolType.PICKBBOX
        self.task = TaskExecutor()
        # Number of discrete rotation angles
        self.rotation_angles = 10
        cv2.namedWindow(self.snapshot_win, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.namedWindow(self.stream_win)
        # highgui function called when mouse events occur
        cv2.setMouseCallback(self.snapshot_win, self.get_coords)

    def cleanup(self):
        """Release resources"""
        cv2.destroyAllWindows()

    def run(self, streamer: CameraStream) -> None:
        """Run GUI"""
        self.handle_stream(streamer)
        if self.snapshot:
            self.handle_snapshot()
        key_press = cv2.waitKey(1) & 0xFF
        self.handle_keypress(streamer, key_press)

    def handle_stream(self, streamer: CameraStream) -> None:
        """Display and interact with stream window"""
        if streamer.rgb is not None and self.stream_type is StreamType.RGB:
            # OpenCV handles bgr format instead of rgb so we convert first
            bgr = cv2.cvtColor(streamer.rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.stream_win, bgr)
        if streamer.depth is not None and self.stream_type is StreamType.DEPTH:
            cv2.imshow(self.stream_win, streamer.depth)

    def handle_snapshot(self) -> None:
        """Display and interact with snapshot"""  
        snap_type = self.stream_type is not StreamType.RGB
        disp_img = self.snapshot[snap_type]
        # convert depth mono channel to bgr so we can display colored bbox
        if self.snapshot[snap_type].ndim == 2:
            disp_img = cv2.cvtColor(disp_img, cv2.COLOR_GRAY2RGB)
        disp_img = cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR)
        # Draw and display image
        rot_color = (0, 0, np.iinfo(disp_img.dtype).max)
        pick_bbox_color = (0, np.iinfo(disp_img.dtype).max, 0)
        place_bbox_color = (np.iinfo(disp_img.dtype).max, 0, 0)
        draw_on_disp_img(disp_img,self.pick_data, pick_bbox_color, rot_color)
        draw_on_disp_img(disp_img,self.place_data, place_bbox_color, rot_color)
        cv2.imshow(self.snapshot_win, disp_img)

    def handle_keypress(self, streamer, key_press) -> None:
        """Handle keypress in gui window"""
        # Stream related controls
        if key_press == ord('a'):
            self.stream_type = StreamType.RGB
        if key_press == ord('d'):
            self.stream_type = StreamType.DEPTH
        if key_press == ord('s'):
            self.snapshot = (streamer.rgb.copy(), streamer.depth.copy())
            rospy.loginfo("Capturing snapshot from camera stream")
        # Snapshot related controls
        if key_press == ord('1'):
            self.pick_data = {'rotation': 0, 'bbox': [], 'rotline': []}
            rospy.loginfo("Clearing up pick data")
        if key_press == ord('2'):
            self.place_data = {'rotation': 0, 'bbox': [], 'rotline': []}
            rospy.loginfo("Clearing up place data")
        if key_press == ord('p'):
            data = self.pick_data
            if len(data['bbox']) == 2 and len(data['rotline']) == 2:
                tinfo = TaskInfo(self.snapshot[0], self.snapshot[1], data['bbox'], data['rotation'])
                rospy.loginfo("Executing pick task")
                self.task.pick(tinfo)
        if key_press == ord('l'):
            data = self.place_data
            if len(data['bbox']) == 2 and len(data['rotline']) == 2:
                tinfo = TaskInfo(self.snapshot[0], self.snapshot[1], data['bbox'], data['rotation'])
                rospy.loginfo("Executing place task")
                self.task.place(tinfo)
        if key_press == ord('h'):
            rospy.loginfo("Executing home task")
            self.task.home()
        if key_press == ord('S'):
            rospy.loginfo("Saving demonstration...")
        if key_press == ord('q'):
            raise KeyboardInterrupt

    def get_coords(self, action, x, y, flags, *userdata) -> None:
        """Mouse callback function which is used to collect bbox coords"""
        # Start marking the pick bbox/rotation
        if action == cv2.EVENT_LBUTTONDOWN:
            self.selected_tool = ToolType.PICKROTATE
            if len(self.pick_data['bbox']) == 0:
                self.selected_tool = ToolType.PICKBBOX
                self.pick_data['bbox'] = [(x,y)]
        # Release and freeze the pick bbox/rotation value
        elif action == cv2.EVENT_LBUTTONUP:
            if self.selected_tool is ToolType.PICKBBOX:
                rospy.loginfo(f"Captured pick bbox coordinates: {self.pick_data['bbox']}")
            if self.selected_tool is ToolType.PICKROTATE:
                rospy.loginfo(f"Captured pick rotation angle: {self.pick_data['rotation']}")
            self.selected_tool = ToolType.NONE
        # Start marking the place bbox/rotation
        elif action == cv2.EVENT_RBUTTONDOWN:
            self.selected_tool = ToolType.PLACEROTATE
            if len(self.place_data['bbox']) == 0:
                self.selected_tool = ToolType.PLACEBBOX
                self.place_data['bbox'] = [(x,y)]
        # Release and freeze the place bbox/rotation value
        elif action == cv2.EVENT_RBUTTONUP:
            if self.selected_tool is ToolType.PLACEBBOX:
                rospy.loginfo(f"Captured place bbox coordinates: {self.place_data['bbox']}")
            if self.selected_tool is ToolType.PLACEROTATE:
                rospy.loginfo(f"Captured place rotation angle: {self.place_data['rotation']}")
            self.selected_tool = ToolType.NONE
        # Assign the pick/place bbox/rotation value based on selected tool
        elif action == cv2.EVENT_MOUSEMOVE:
            # Handle bbox
            if len(self.pick_data['bbox']) > 0 and self.selected_tool is ToolType.PICKBBOX:
                self.pick_data['bbox'] = [self.pick_data['bbox'][0], (x, y)]
            if len(self.place_data['bbox']) > 0 and self.selected_tool is ToolType.PLACEBBOX:
                self.place_data['bbox'] = [self.place_data['bbox'][0], (x, y)]
            # Handle rotation
            data = None
            if self.selected_tool is ToolType.PICKROTATE:
                data = self.pick_data
            if self.selected_tool is ToolType.PLACEROTATE:
                data = self.place_data
            if data:
                line, theta = get_line_theta(data['bbox'], (x,y))
                data['rotation'] = int(theta // self.rotation_angles)
                data['rotline'] = line
