"""Utility functions"""
import numpy as np
import cv2

def get_origin_from_bbox(bbox):
    """Interpret origin from bounding box coordinates"""
    boxw, boxh = abs(bbox[0][0] - bbox[1][0]), abs(bbox[0][1] - bbox[1][1])
    origin = (np.min([bbox[0][0], bbox[1][0]]) + boxw//2,
                np.min([bbox[0][1], bbox[1][1]]) + boxh//2)
    return origin, boxw, boxh

def get_line_theta(bbox, cursor):
    """Get line to draw and appropriate theta value (rotation angle)"""
    origin, boxw, boxh = get_origin_from_bbox(bbox)
    # we mirror the y-axis because value of it is 0 at the top
    perpendicular = -(cursor[1] - origin[1])
    base = cursor[0] - origin[0]
    # Get theta
    theta = np.arctan2(perpendicular, base)
    # Get line
    radius = np.sqrt(boxw**2 + boxh**2)/2
    px = int(radius*np.cos(theta + np.pi)) + origin[0]
    py = int(radius*np.sin(theta + np.pi)) + origin[1]
    px2 = int(radius*np.cos(theta)) + origin[0]
    py2 = int(radius*np.sin(theta)) + origin[1]
    # We mirror the drawn line along X-Axis so that the line follows the cursor
    line = ((px, py2), (px2, py))
    theta = theta*180/np.pi
    if theta < 0:
        theta = theta + 360
    return line, theta

def draw_on_disp_img(img, data, bbox_color, rot_color) -> None:
    """Draw on display image of snapshot given pick/place data"""
    # Draw bbox
    if len(data['bbox']) == 2:
        cv2.rectangle(img, data['bbox'][0], data['bbox'][1], bbox_color, 2, 8)
        # Draw rotation line
        if len(data['rotline']) == 2:
            cv2.line(img, data['rotline'][0], data['rotline'][1], rot_color, 2)
            cv2.circle(img, data['rotline'][1], 5, rot_color, 2)
            # Draw rotation angle text
            text = f"K={data['rotation']}"
            origin, width, height = get_origin_from_bbox(data['bbox'])
            tx = origin[0] - width//2
            ty = np.max([0, origin[1] - height//2 - 10])
            cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2, cv2.LINE_AA)
