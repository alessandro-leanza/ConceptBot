#!/usr/bin/env python

import numpy as np
import pyrealsense2 as rs
import cv2
from ultralytics import YOLO
import supervision as sv
import rospy
from vision.msg import RobotState
from scipy.spatial.transform import Rotation as R
from autolab_core import RigidTransform
import time

# Global variables for transformation
O_T_C = np.zeros((4, 4))

def callback(data):
    """
    Callback to update the transformation matrix O_T_C
    """
    global O_T_C
    # Definisci la matrice estrinseca EE_T_C (End Effector to Camera)
    EE_T_C_cmarray = np.array([
        0.022288845304241245, 0.9997492180952425, -0.002169860123366514, 0.0,
        -0.9996630147287806, 0.02231570617364864, 0.013261456983954008, 0.0,
        0.013306553211462676, 0.00187345463492401318, 0.9999097086565906, 0.0,
        0.02711385979847173, -0.03393779506264873, -0.037530544916826725, 1.0
    ])
    EE_T_C = np.array(EE_T_C_cmarray, order='F').reshape((4,4), order='F')

    # Extract the O_T_EE transformation from the ROS message.
    O_T_EE_cmarray = np.array(data.O_T_EE)
    O_T_EE = np.array(O_T_EE_cmarray, order='F').reshape((4,4), order='F')

    # Calculate the transformation O_T_C
    O_T_C = np.matmul(O_T_EE, EE_T_C)

def KG_YOLO():
    """
    Main function that performs object detection and returns
    found_objects and object_centers.

    Returns:
        found_objects (list): List of detected object names.
        object_centers (list): List of 3D coordinates of the objects referred to the robot.
    """
    global O_T_C

    # Initialize the ROS node if it has not already been initialized
    # if not rospy.core.is_initialized():
    #     rospy.init_node('kg_yolo_node', anonymous=True)

    # Subscription to ROS topic to receive robot status (communication my ros node to topic)
    rospy.Subscriber('/robot_state_publisher_node_1/robot_state', RobotState, callback)

    # Initialize the RealSense camera pipeline.
    pipeline = rs.pipeline()
    config = rs.config()
    width = 1280
    height = 720
    fps = 30
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    pipeline.start(config)

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    # Alignment of depth frames to color frame
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Wait for the O_T_C matrix to be updated
    timeout = 5  # sec
    start_time = time.time()
    while np.all(O_T_C == 0) and (time.time() - start_time) < timeout:
        rospy.sleep(0.1)
    if np.all(O_T_C == 0):
        rospy.logerr("Timeout: The transformation was not received O_T_C.")
        pipeline.stop()
        return [], []

    # Upload YOLO model
    model = YOLO("") ###

    # Acquire frames
    for i in range(20):
            pipeline.wait_for_frames()
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        depth_frame = depth_frame

        color_image = np.asanyarray(color_frame.get_data())
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

        depth_color_frame = rs.colorizer().colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_cvt = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # YOLO results
        results = model(color_image)[0]
        detections = sv.Detections.from_ultralytics(results)

        oriented_box_annotator = sv.OrientedBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        annotated_frame = oriented_box_annotator.annotate(
            scene=color_image, detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections)

        context = ''
        sv.plot_image(image=annotated_frame, size=(16, 16))
        
        found_objects = []
        object_centers = []

        for detection_idx in range(len(detections)):
            #object_name = detection.class_name
            object_name = detections['class_name'][detection_idx]
            object_name_str = str(object_name)
            print('ciao3')
        
            # Get the coordinates of the oriented bounding box
            box = results[detection_idx].obb.xywhr

            # Get the distance to the depth frame
            vdist = depth_frame.get_distance(box[0][0], box[0][1])

            # Deprojects the pixel into 3D coordinates in the camera frame
            point = rs.rs2_deproject_pixel_to_point(color_intrin, [box[0][0], box[0][1]], vdist)

            print(object_name_str, point[0], point[1], point[2])

            # Calculate the center of the bounding box
            center_x = box[0][0]
            center_y = box[0][1]

            # Get the distance to the depth frame
            vdist = depth_frame.get_distance(center_x, center_y)
            if vdist == 0:
                continue 

            # Deprojects the pixel into 3D coordinates in the camera frame
            point_camera = rs.rs2_deproject_pixel_to_point(color_intrin, [center_x, center_y], vdist)
            point_camera = np.array([point_camera[0], point_camera[1], point_camera[2], 1.0]).reshape((4,1))

            rotation_matrix = O_T_C[0:3, 0:3]
            C_T_o_cmarray = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, point_camera[0], point_camera[1], point_camera[2], 1.0])
            C_T_o = np.array(C_T_o_cmarray, order='F').reshape((4,4), order='F')
            O_T_o = np.matmul(O_T_C, C_T_o)
            x_g = O_T_o[0, 3]
            y_g = O_T_o[1, 3]
            z_g = O_T_o[2, 3]
            #point_robot = [x_g, y_g, z_g]

            print(object_name_str, x_g, y_g, z_g)
            print('ciao4')

            # Transforms coordinates in the robot's reference system
            #point_robot = np.matmul(O_T_C, point_camera)
            #x_g = point_robot[0, 0]
            #y_g = point_robot[1, 0]
            #z_g = point_robot[2, 0]

            # Add to list if coordinates are valid
            if not (x_g == 0 and y_g == 0 and z_g == 0):
                found_objects.append(object_name_str)
                object_centers.append([object_name_str, x_g, y_g, z_g])

        found_objects.append('user')
        object_centers.append(['user', 0.500, 0.000, 0.201])

        found_objects.append('side-plate')
        object_centers.append(['side-plate', 0.32553, 0.017014, -0.021144])


        break

    pipeline.stop()

    return found_objects, object_centers
