"""Logic for executing task on franka panda"""
from typing import List, Any
from dataclasses import dataclass
from enum import Enum
import copy

import numpy as np
import actionlib
import rospy
import moveit_commander
import geometry_msgs.msg
import tf

import franka_gripper.msg
from moveit_msgs.msg import RobotTrajectory
from actionlib_msgs.msg import GoalStatusArray

from cliport_label.utils import get_avg_3d_centroid, get_relative_orientation
from moveit_msgs.msg import Constraints, OrientationConstraint

@dataclass
class TaskInfo:
    img_rgb: np.ndarray
    img_depth: np.ndarray
    bbox: List[Any]
    rotation: int


class GoalStatus(Enum):
    PENDING = 0   # The goal has yet to be processed by the action server
    ACTIVE  = 1   # The goal is currently being processed by the action server
    PREEMPTED  = 2   # The goal received a cancel request after it started executing
                     #   and has since completed its execution (Terminal State)
    SUCCEEDED = 3   # The goal was achieved successfully by the action server (Terminal State)
    ABORTED = 4   # The goal was aborted during execution by the action server due
                  #    to some failure (Terminal State)
    REJECTED = 5   # The goal was rejected by the action server without being processed,
                   #    because the goal was unattainable or invalid (Terminal State)
    PREEMPTING = 6   # The goal received a cancel request after it started executing
                     #    and has not yet completed execution
    RECALLING = 7   # The goal received a cancel request before it started executing,
                    #    but the action server has not yet confirmed that the goal is canceled
    RECALLED = 8   # The goal received a cancel request before it started executing
                   #    and was successfully cancelled (Terminal State)
    LOST = 9   # An action client can determine that a goal is LOST. This should not be
                #    sent over the wire by an action server


class TaskExecutor:

    def __init__(self, config) -> None:
        """Initialize stuff"""
        self.config = config
        moveit_commander.roscpp_initialize([''])
        # initialize moveit commander
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        # Set grasp tool as EE link
        self.move_group.set_end_effector_link("panda_hand_tcp")
        self.subscriber_feedback = rospy.Subscriber("/move_group/status",
            GoalStatusArray, self.feedback_callback,)
        self.move_group_status = GoalStatus.PENDING
        # Clients to send commands to the gripper
        self.grasp_action_client = actionlib.SimpleActionClient("/franka_gripper/grasp", franka_gripper.msg.GraspAction)
        self.move_action_client = actionlib.SimpleActionClient("/franka_gripper/move", franka_gripper.msg.MoveAction)
        # Rotation angles. yaw angle is given by yaw=rotation_angle*K [k = 0,35]
        self.rotation_angles = 10
        # Transformation Matrices
        # Pixel to Camera coordinate system
        self.intrinsic = [
            [609.9600830078125, 0.0, 336.7248229980469,],
            [0.0, 609.9955444335938, 249.56271362304688],
            [0.0, 0.0, 1.0]
          ]
        #aligned_depth_to_color_frame
        self.listener = tf.TransformListener()
        self.listener.waitForTransform("/panda_link0", "/camera_color_optical_frame", rospy.Time(0), rospy.Duration(4.0))
        transform = self.listener.lookupTransform("/panda_link0",
                                   "/camera_color_optical_frame", #target frame
                                   rospy.Time(0),) #get the tf at first available time
        # Camera to end effector transform 
        self.extrinsic = {
            'xyz': transform[0],
            'quaternion': transform[1]
        } 
        rospy.loginfo(f"Camera intrinsic: {self.intrinsic}")
        rospy.loginfo(f"Camera-to-base extrinsic: {self.extrinsic}")
        # Bring robot to home position during initialization
        self.home(wait=True)
        self.default_ee_pose = self.move_group.get_current_pose()
        rospy.loginfo(f"End effector pose at home location: {self.default_ee_pose}")
        

    def init_path_constraints(self):
        self.path_constraints = Constraints()
        self.path_constraints.name = "yawonly"
        orientation_constraint = OrientationConstraint()
        ee_link = self.move_group.get_end_effector_link()
        pose = self.move_group.get_current_pose(ee_link)
        orientation_constraint.header = pose.header
        orientation_constraint.link_name = ee_link
        orientation_constraint.orientation = pose.pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.1
        orientation_constraint.absolute_y_axis_tolerance = 0.1
        orientation_constraint.absolute_z_axis_tolerance = 3.14
        orientation_constraint.weight = 1

        self.path_constraints.orientation_constraints.append(orientation_constraint)
        # self.move_group.set_path_constraints(self.path_constraints)

    def enable_path_constraints(self):
        self.move_group.set_path_constraints(self.path_constraints)

    def disable_path_constraints(self):
        self.move_group.set_path_constraints(None)

    def pick(self, data: TaskInfo):
        """Execute pick task"""
        # This is used to execute up-down movement when grasping the target
        z_offset = 0.035
        # Clear existing execution and start pick task from home
        self.stop()
        self.home()
        # Get target position and orientation
        target_xyz, camera_xyz = get_avg_3d_centroid(data.img_depth, data.bbox, self.intrinsic, self.extrinsic)
        rospy.loginfo(f"{camera_xyz = }")
        rospy.loginfo(f"{target_xyz = }")
        ee_wxyz = [self.default_ee_pose.pose.orientation.w,
                    self.default_ee_pose.pose.orientation.x,
                    self.default_ee_pose.pose.orientation.y,
                    self.default_ee_pose.pose.orientation.z]
        target_wxyz = get_relative_orientation(ee_wxyz, data.rotation*self.rotation_angles)

        pose = geometry_msgs.msg.Pose()
        pose.position.x = target_xyz[0]
        pose.position.y = target_xyz[1]
        pose.position.z = target_xyz[2]
        pose.orientation.w = target_wxyz[0]
        pose.orientation.x = target_wxyz[1]
        pose.orientation.y = target_wxyz[2]
        pose.orientation.z = target_wxyz[3]

        # Move above object and open gripper
        rospy.loginfo("Moving towards pick object and opening gripper")
        pose_up = copy.deepcopy(pose)
        pose_up.position.z += z_offset
        self.execute_cartesian_path([pose_up])
        self.open_gripper()
        # Move down and grasp object
        rospy.loginfo("Moving down and grasping pick object")
        pose_down = copy.deepcopy(pose)
        pose_down.position.z -= z_offset
        self.execute_cartesian_path([pose_down])
        self.close_gripper()
        # Move up again
        rospy.loginfo("Moving up again after picking object")
        self.execute_cartesian_path([pose_up])
    
    def place(self, data: TaskInfo):
        """Execute place task"""
        # This is used to execute up-down movement when grasping the target
        z_offset = 0.035
        # Clear existing execution and start pick task from home
        self.stop()
        self.home()
        # Get target position and orientation
        target_xyz, camera_xyz = get_avg_3d_centroid(data.img_depth, data.bbox, self.intrinsic, self.extrinsic)
        rospy.loginfo(f"{camera_xyz = }")
        rospy.loginfo(f"{target_xyz = }")
        ee_wxyz = [self.default_ee_pose.pose.orientation.w,
                    self.default_ee_pose.pose.orientation.x,
                    self.default_ee_pose.pose.orientation.y,
                    self.default_ee_pose.pose.orientation.z]
        target_wxyz = get_relative_orientation(ee_wxyz, data.rotation*self.rotation_angles)

        pose = geometry_msgs.msg.Pose()
        pose.position.x = target_xyz[0]
        pose.position.y = target_xyz[1]
        pose.position.z = target_xyz[2]
        pose.orientation.w = target_wxyz[0]
        pose.orientation.x = target_wxyz[1]
        pose.orientation.y = target_wxyz[2]
        pose.orientation.z = target_wxyz[3]

        # Move above object and open gripper
        rospy.loginfo("Moving towards place object and opening gripper")
        pose_up = copy.deepcopy(pose)
        pose_up.position.z += z_offset
        self.execute_cartesian_path([pose_up])
        self.open_gripper()


    def feedback_callback(self, data):
        """Callback function for topic /move_group/feedback"""
        # Only take the latest status
        if len(data.status_list) > 0:
            self.move_group_status = GoalStatus(data.status_list[-1].status)

    def home(self, wait=True):
        """Goto home position"""
        # Clear existing pose targets
        self.move_group.clear_pose_targets()
        # Plan home joint values
        home_joint_values = [-0.10978979745454956, -0.7703535289764404, -0.05097640468462238, -2.3268556568809795, 0.0010342414430801817, 1.5708663142522175, 0.7840747220798833]
        self.move_group.set_joint_value_target(home_joint_values)
        plan = self.move_group.plan()
        self.execute_plan(plan, wait)
        self.init_path_constraints()


    def execute_cartesian_path(self, waypoints):
        """Execute cartesian path with some safety checks regarding pose waypoints"""
        z_min, z_max = 0.02, 0.30
        for pose in waypoints:
            if pose.position.z < z_min:
                rospy.logwarn(f"{pose.position.z = } is invalid. Using {z_min} instead")
                pose.position.z = z_min
            if pose.position.z > z_max:
                rospy.logwarn(f"{pose.position.z = } is invalid. Using {z_max} instead")
                pose.position.z = z_max
        plan, _ = self.move_group.compute_cartesian_path(waypoints, 0.01, 0.0, path_constraints=self.path_constraints)  # jump_threshold
        self.execute_plan(plan)
        

    def execute_plan(self, plan, wait=True) -> None:
        """Execute a given plan through move group"""
        rospy.loginfo(f"Current move group status: {self.move_group_status}")
        if self.move_group_status == GoalStatus.ACTIVE:
                rospy.logwarn("Robot busy, another trajectory is being executed. Wait for it to finish")
                return
        
        if isinstance(plan, RobotTrajectory):
            plan = [True, plan]
        if plan[0]:
            self.move_group.execute(plan[1], wait=wait)
        else:
            rospy.logwarn("Could not plan trajectory from current pose to home pose")

    def stop(self) -> None:
        """Stop execution"""
        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def cleanup(self) -> None:
        """Cleanup task executor"""
        self.stop()
        moveit_commander.roscpp_shutdown()

    def open_gripper(self) -> None:
        """Open gripper"""
        if not self.config["taskexecutor"]["enable_gripper"]:
            return
        goal = franka_gripper.msg.MoveGoal()
        goal.width = 0.08
        goal.speed = 0.1
        self.move_action_client.send_goal(goal)
        self.move_action_client.wait_for_result()


    def close_gripper(self):
        """Grasp object by closing gripper"""
        if not self.config["taskexecutor"]["enable_gripper"]:
            return
        goal = franka_gripper.msg.GraspGoal()
        goal.width = 0.00
        goal.speed = 0.1
        goal.force = 5  # limits 0.01 - 50 N
        goal.epsilon = franka_gripper.msg.GraspEpsilon(inner=0.08, outer=0.08)
        self.grasp_action_client.send_goal(goal)
        self.grasp_action_client.wait_for_result()
