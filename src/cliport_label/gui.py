"""GUI for label tool"""
from enum import Enum

import cv2
import numpy as np
import rospy

from cliport_label.taskexecutor import TaskInfo
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

    def __init__(self, config, streamer, taskexecutor) -> None:
        """Initialize stuff"""
        self.stream_win = "stream"
        self.snapshot_win = "snapped_observation"
        self.stream_type = StreamType.RGB
        self.snapshot = None
        self.pick_data = {'rotation': 0, 'bbox': [], 'rotline': []}
        self.place_data = {'rotation': 0, 'bbox': [], 'rotline': []}
        self.selected_tool = ToolType.PICKBBOX
        self.task = taskexecutor
        self.streamer = streamer
        self.controls = config["controlkeys"]
        # Number of discrete rotation angles
        self.rotation_angles = 10
        if self.task is not None:
            self.task.rotation_angles = self.rotation_angles
        cv2.namedWindow(self.snapshot_win, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.namedWindow(self.stream_win)
        # highgui function called when mouse events occur
        cv2.setMouseCallback(self.snapshot_win, self.get_coords)

    def cleanup(self):
        """Release resources"""
        cv2.destroyAllWindows()
        if self.task:
            self.task.cleanup()

    def run(self) -> None:
        """Run GUI"""
        self.handle_stream()
        if self.snapshot:
            self.handle_snapshot()
        key_press = cv2.waitKey(1) & 0xFF
        self.handle_keypress(key_press)

    def depth_to_heatmap(self, depth) -> None:
        """Normalize depth image to color heatmap for display"""
        valid = (depth != 0)
        ranged = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) # dmin -> 0.0, dmax -> 1.0
        ranged[ranged < 0] = 0 # saturate
        ranged[ranged > 1] = 1
        output = 1.0 - ranged # 0 -> white, 1 -> black
        output[~valid] = 0 # black out invalid
        output **= 1/2.2 # most picture data is gamma-compressed
        output = cv2.normalize(output, output, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        output = cv2.applyColorMap(output, cv2.COLORMAP_JET)
        
        return output

    def handle_stream(self) -> None:
        """Display and interact with stream window"""
        if self.streamer.rgb is not None and self.stream_type is StreamType.RGB:
            # OpenCV handles bgr format instead of rgb so we convert first
            bgr = cv2.cvtColor(self.streamer.rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.stream_win, bgr)
        if self.streamer.depth is not None and self.stream_type is StreamType.DEPTH:
            cv2.imshow(self.stream_win, self.depth_to_heatmap(self.streamer.depth))

    def handle_snapshot(self) -> None:
        """Display and interact with snapshot"""  
        snap_type = self.stream_type is not StreamType.RGB
        disp_img = self.snapshot[snap_type]
        # convert mono depth to color heatmap before displaying
        if self.snapshot[snap_type].ndim == 2:
            disp_img = self.depth_to_heatmap(disp_img)
        else:
            disp_img = cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR)
        # Draw and display image
        rot_color = (0, 0, np.iinfo(disp_img.dtype).max)
        pick_bbox_color = (0, np.iinfo(disp_img.dtype).max, 0)
        place_bbox_color = (np.iinfo(disp_img.dtype).max, 0, 0)
        draw_on_disp_img(disp_img,self.pick_data, pick_bbox_color, rot_color)
        draw_on_disp_img(disp_img,self.place_data, place_bbox_color, rot_color)
        cv2.imshow(self.snapshot_win, disp_img)

    def handle_keypress(self, key_press) -> None:
        """Handle keypress in gui window"""
        # Stream related controls
        if key_press == ord(self.controls['rgb']):
            self.stream_type = StreamType.RGB
        if key_press == ord(self.controls['depth']):
            self.stream_type = StreamType.DEPTH
        if key_press == ord(self.controls['snapshot']):
            self.snapshot = (self.streamer.rgb.copy(), self.streamer.depth.copy())
            rospy.loginfo("Capturing snapshot from camera stream")
        # Snapshot related controls
        if key_press == ord(self.controls['clear_pick']):
            self.pick_data = {'rotation': 0, 'bbox': [], 'rotline': []}
            rospy.loginfo("Clearing up pick data")
        if key_press == ord(self.controls['clear_place']):
            self.place_data = {'rotation': 0, 'bbox': [], 'rotline': []}
            rospy.loginfo("Clearing up place data")
        if key_press == ord(self.controls['save']):
            rospy.loginfo("Saving demonstration...")
        if key_press == ord(self.controls['quit']):
            raise KeyboardInterrupt
        # Ignore task executor related commands if its set to None
        if self.task is not None:
            if key_press == ord(self.controls['pick']):
                data = self.pick_data
                if len(data['bbox']) == 2 and len(data['rotline']) == 2:
                    tinfo = TaskInfo(self.snapshot[0], self.snapshot[1], data['bbox'], data['rotation'])
                    rospy.loginfo("Executing pick task")
                    self.task.pick(tinfo)
            if key_press == ord(self.controls['place']):
                data = self.place_data
                if len(data['bbox']) == 2 and len(data['rotline']) == 2:
                    tinfo = TaskInfo(self.snapshot[0], self.snapshot[1], data['bbox'], data['rotation'])
                    rospy.loginfo("Executing place task")
                    self.task.place(tinfo)
            if key_press == ord(self.controls['open_gripper']):
                rospy.loginfo("Opening gripper")
                self.task.open_gripper()
            if key_press == ord(self.controls['close_gripper']):
                rospy.loginfo("Closing gripper")
                self.task.close_gripper()
            if key_press == ord(self.controls['home']):
                rospy.loginfo("Executing home task")
                self.task.home()
            if key_press == ord(self.controls['stop_execution']):
                rospy.loginfo("Stopping all execution")
                self.task.stop()

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
