#!/usr/bin/env python

import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm
import time
import rospy
from std_msgs.msg import String

class PickAndPlace:
    def __init__(self):
        """
        Initialize the Franka Arm robotic arm instance.
        """
        try:
            self.fa = FrankaArm(with_gripper=True)
            print("FrankaArm successfully initialized.")
        except Exception as e:
            print(f"Error in initializing FrankaArm: {e}")
            self.fa = None

        self.rest_position = [0.4, 0.0, 0.5]  # [x, y, z] in m
        self.rotation_matrix = [[1, 0, 0],
                                [0, -1, 0],
                                [0, 0, -1]] 

    def pick_object(self, object_name, object_coords):
        """
        Performs the pick operation for a specific object.

        Args:
            object_name (str): Name of the object to be picked.
            object_coords (list or np.array): Coordinates [x, y, z] of the object referenced by the robot.
        """

        print(f"Resetting joints for {object_name}...")
        self.fa.reset_joints()
        #self.fa.goto_gripper()

        pick_position = [
            object_coords[0],
            object_coords[1],
            object_coords[2] + 0.05
        ]

        print(f"Calculating pick position for {object_name}: {pick_position}")

        pick_transform = RigidTransform(
            translation=pick_position,
            rotation=self.rotation_matrix,
            from_frame="franka_tool",
            to_frame="world"
        )

        current_pose = self.fa.get_pose().translation
        distance = np.linalg.norm(np.array(current_pose) - np.array(pick_position))
        duration = float(distance * 12)

        print(f"Moving above the pick position for {object_name}... Duration: {duration} seconds")
        self.fa.goto_pose(
            tool_pose=pick_transform,
            duration=duration,
            use_impedance=False,
            ignore_virtual_walls=True,
            block=True
        )

        print(f"Gripper positioned 5 cm above {object_name}.")




    def closing_gripper(self, object_name, object_coords, objects_info):
        """
        Gripper closure process composed of three primitives:
        - Descent of the arm by 5 cm to reach the exact pick position
        - Gripper closure given a certain modifiable width and force
        - Ascent of the arm by 10 cm from the pick position to then facilitate the place operation

        """
        pick_position = [
            object_coords[0],
            object_coords[1],
            object_coords[2]
        ]

        print(f"Calculating pick position for {object_name}: {pick_position}")

        final_pick_transform = RigidTransform(
            translation=pick_position,
            rotation=self.rotation_matrix,
            from_frame="franka_tool",
            to_frame="world"
        )
        
        current_pose = self.fa.get_pose().translation
        distance = np.linalg.norm(np.array(current_pose) - np.array(pick_position))
        duration = float(distance * 18)  

        print(f"Moving above the pick position for {object_name}... Duration: {duration} seconds")
        self.fa.goto_pose(
            tool_pose=final_pick_transform,
            duration=duration,
            use_impedance=False,
            ignore_virtual_walls=True,
            block=True
        )

        print(f"Gripper ready to pick the {object_name}.")

        
        gripper_width = 0.02 # m
        gripper_force = 10.0 # N

        if object_name == "Obj1":
            gripper_width = 0.02
            gripper_force = 10.0
        elif object_name == "Obj2":
            gripper_width = 0.02
            gripper_force = 10.0
        elif object_name == "Obj3":
            gripper_width = 0.02
            gripper_force = 10.0

        if object_name in objects_info:
            properties = objects_info[object_name]
            if properties.get("fragile") == "Yes":
                gripper_width = 0.02  
                gripper_force = 7.0    
            elif properties.get("hold liquid") == "Yes":
                gripper_width = 0.02  
                gripper_force = 10.0

        print(f"Closing gripper to width {gripper_width} m, with force {gripper_force} N ...")

        #self.fa.goto_gripper(width = gripper_width, force = gripper_force)
        self.fa.close_gripper()

        print(f"Gripper Closed")

        time.sleep(1)

        after_pick_position = [
            object_coords[0],
            object_coords[1],
            object_coords[2] + 0.15
        ]

        after_pick_transform = RigidTransform(
            translation=after_pick_position,
            rotation=self.rotation_matrix,
            from_frame="franka_tool",
            to_frame="world"
        )

        current_pose = self.fa.get_pose().translation
        distance = np.linalg.norm(np.array(current_pose) - np.array(after_pick_position))
        duration = float(distance * 12)  

        print(f"Moving above the pick position for {object_name}... Duration: {duration} seconds")
        self.fa.goto_pose(
            tool_pose=after_pick_transform,
            duration=duration,
            use_impedance=False,
            ignore_virtual_walls=True,
            block=True
        )

        print(f"Gripper Closing process finished.")


    def place_object(self, place_location, place_coords):
        """
        Performs the place operation for a specific object.

        Args:
            place_location (str): Name of the destination of place.
            place_coords (list or np.array): Coordinates [x, y, z] of the place destination referenced by the robot.
        """
        place_position = [
            place_coords[0],
            place_coords[1],
            place_coords[2] + 0.1
        ]

        print(f"Calculating place position for {place_location}: {place_position}")

        place_transform = RigidTransform(
            translation=place_position,
            rotation=self.rotation_matrix,
            from_frame="franka_tool",
            to_frame="world"
        )

        current_pose = self.fa.get_pose().translation
        distance = np.linalg.norm(np.array(current_pose) - np.array(place_position))
        duration = float(distance * 18)

        print(f"Moving to place position at {place_location}... Duration: {duration} seconds")
        self.fa.goto_pose(
            tool_pose=place_transform,
            duration=duration,
            use_impedance=False,
            ignore_virtual_walls=True,
            block=True
        )

        print(f"Gripper positioned 5 cm above {place_location}.")


    def opening_gripper(self, place_location, place_coords):
        """
        Gripper opening process composed of three primitives:
        - Descent of the arm by 5 cm to reach the exact place position
        - Opening of the gripper
        - Ascent of the arm 10 cm from the place position to then facilitate the rest operation
        
        """
        place_position = [
            place_coords[0],
            place_coords[1],
            place_coords[2] + 0.11
        ]

        print(f"Calculating place position for {place_location}: {place_position}")

        final_place_transform = RigidTransform(
            translation=place_position,
            rotation=self.rotation_matrix,
            from_frame="franka_tool",
            to_frame="world"
        )

        current_pose = self.fa.get_pose().translation
        distance = np.linalg.norm(np.array(current_pose) - np.array(place_position))
        duration = float(distance * 12)  

        print(f"Moving above the place position {place_location}... Duration: {duration} seconds")
        self.fa.goto_pose(
            tool_pose=final_place_transform,
            duration=duration,
            use_impedance=False,
            ignore_virtual_walls=True,
            block=True
        )

        print(f"Gripper ready to place the object at the {place_location}.")
     
    def opening_gripper_2(self, place_location, place_coords):
        after_place_position = [
            place_coords[0],
            place_coords[1],
            place_coords[2] + 0.1
        ]

        after_place_transform = RigidTransform(
            translation=after_place_position,
            rotation=self.rotation_matrix,
            from_frame="franka_tool",
            to_frame="world"
        )

        current_pose = self.fa.get_pose().translation
        distance = np.linalg.norm(np.array(current_pose) - np.array(after_place_position))
        duration = float(distance * 18)  

        print(f"Moving above the place position {place_location}... Duration: {duration} seconds")
        self.fa.goto_pose(
            tool_pose=after_place_transform,
            duration=duration,
            use_impedance=False,
            ignore_virtual_walls=True,
            block=True
        )

        print(f"Gripper Opening process finished.")



    def return_to_rest(self):
        """
        It moves the robotic arm to the rest position.
        """

        rest_transform = RigidTransform(
            translation=self.rest_position,
            rotation=self.rotation_matrix,
            from_frame="franka_tool",
            to_frame="world"
        )

        current_pose = self.fa.get_pose().translation
        distance = np.linalg.norm(np.array(current_pose) - np.array(self.rest_position))
        duration = float(distance * 24) 

        print(f"Moving to rest position... Duration: {duration} seconds")
        self.fa.goto_pose(
            tool_pose=rest_transform,
            duration=duration,
            use_impedance=False,
            ignore_virtual_walls=True,
            block=True
        )

        self.fa.open_gripper()

        print("Robot moved to rest position.")


