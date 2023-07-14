import cv2 
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from collections import defaultdict

from HeadPoseEstimator import HeadPoseEstimator
from pose import angle_vector, Pose

# BGR color
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
RED_COLOR = (0, 0, 255)
PURPLE_COLOR = (255, 0, 255)
ORANGE_COLOR = (0, 128, 255)
WHITE_COLOR = (255, 255, 255)
YELLOW_COLOR = (0, 255, 255)
CYAN_COLOR = (255, 255, 0)

# Displaying the image
def display(img) :
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

def getSkeleton():
    # body: 1 - 16
    # foot: 17 - 22
    # face: 23 - 90
    # hand: 91 - 132
    skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], # blue
                [11, 12], [5, 11], [6, 12], # purple
                [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], # orange
                [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], # green 
                [15, 17], [15, 18],[15, 19], [16, 20], [16, 21], [16, 22], # green
                [91, 92], [92, 93], [93, 94], [94, 95], [91, 96], [96, 97], 
                [97, 98], [98, 99], [91, 100], [100, 101], [101, 102],
                [102, 103], [91, 104], [104, 105], [105, 106],
                [106, 107], [91, 108], [108, 109], [109, 110],
                [110, 111], [112, 113], [113, 114], [114, 115],
                [115, 116], [112, 117], [117, 118], [118, 119],
                [119, 120], [112, 121], [121, 122], [122, 123],
                [123, 124], [112, 125], [125, 126], [126, 127],
                [127, 128], [112, 129], [129, 130], [130, 131],
                [131, 132]]
    
    return skeleton

def getColorFromIndexSkeleton(index):
    if index < 4:
        return BLUE_COLOR
    if index < 7:
        return PURPLE_COLOR
    if index < 12:
        return ORANGE_COLOR
    return GREEN_COLOR
    
    

HEAD_POSE_HISTORY_LIMIT = 4
FLIP_ANGLE_THRESHOLD = math.pi * 3/4 # Angle of current and previous head pose that will trigger a flip (to prevent noise)

def drawline(image, box1, box2, text):
    # Extract coordinates from bounding boxes
    x1, y1 = box1['tl_coord2d']
    x2, y2 = box2['tl_coord2d']
    # Calculate the centers of the bounding boxes
    center1 = (x1 + (box1['br_coord2d'][0] - x1) // 2, y1 + (box1['br_coord2d'][1] - y1) // 2)
    center2 = (x2 + (box2['br_coord2d'][0] - x2) // 2, y2 + (box2['br_coord2d'][1] - y2) // 2)

    # Draw line between the centers of the two bounding boxes
    line_color = RED_COLOR  # Red color (BGR format)
    line_thickness = 2
    image = cv2.line(image, center1, center2, line_color, line_thickness)

    # Calculate the position to place the text
    text_position = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2 - 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_color = RED_COLOR  # Red color (BGR format)
    text_thickness = 2
    image = cv2.putText(image, text, text_position, font, font_scale, text_color, text_thickness)

    return image

def change_opacity(img, opacity=0.5):
    img = (img * opacity).astype(np.uint8)
    return img

def visualize(args):
    pose_path = args.pose_path
    save_path = args.output_path
    head_pose_history = defaultdict(lambda: [])

    if args.student_staff_path is not None:
        with open(args.student_staff_path) as file: 
            student_staff_data = json.load(file)

    with open(pose_path) as fpose:
        if args.action_path is None:
            faction = None
        else:
            faction = open(args.action_path)
        
        for idx, line in enumerate(fpose):
            pose_data = json.loads(line)
            if faction is not None:
                action_line = faction.readline()
                action_data = json.loads(action_line)

            img_h, img_w = pose_data["img_h"], pose_data["img_w"]

            # print(img_h, img_w)

            if idx == 0:
                out = cv2.VideoWriter(
                    save_path,
                    cv2.VideoWriter_fourcc(*'MJPG'), 
                    15, (img_w, img_h))
                if args.background_path is None or args.background_path == "blank":
                    background = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                    print("blank")
                else:
                    background_img = cv2.imread(args.background_path)
                    background = cv2.resize(background_img, (img_w, img_h))
                    background = change_opacity(background, args.background_opacity)
                continue

            
            img = background.copy()
            pose_data, approach_data = pose_data["pose"], pose_data["approach"]

            pose = pose_data

            head_pose_estimator = HeadPoseEstimator(img_size=(img_h, img_w))

            head_pose = {
                person_id: None 
                for person_id in pose["persons"] 
            }

            # Extract face keypoints to marks
            for person_id in pose["persons"]:
                person_marks = []
                person_keypoints = pose["persons"][person_id]

                # Extract 68 keypoints
                for keypoint_id in range(23, 91):
                    person_marks.append(person_keypoints[str(keypoint_id)][:-1])

                person_marks = np.array(person_marks).astype("float32")
                person_marks[:,0] *= img_w
                person_marks[:,1] *= img_h

                # Try pose estimation with 68 points.
                # Return result is rotation and translation vector of head from the camera
                person_head_pose = head_pose_estimator.solve_pose_by_68_points(
                    person_marks
                )
                
                head_pose[person_id] = person_head_pose

                # gazing_point_2d is a ndarray [p1, p2], 
                # where p1 is the origin coordinate and p2 is the other point coordinate
                gazing_point_2d = head_pose_estimator.estimate_gazing_line(
                    person_head_pose[0], 
                    person_head_pose[1]
                )
                color = GREEN_COLOR

                # If 
                #   - cannot detect head pose (value is zero) or
                #   - head pose change significantly ~pi (probably noise)
                # assign it to the previous pose
                gaze_vector = gazing_point_2d[1]-gazing_point_2d[0]
                if len(head_pose_history[person_id]) > 0:
                    last_head_pose = head_pose_history[person_id][-1]
                    last_gaze_vector = last_head_pose[1]-last_head_pose[0]
                    if np.sum(np.abs(gaze_vector)) < 1e-6:
                        gazing_point_2d = last_head_pose
                        color = YELLOW_COLOR
                    elif angle_vector(
                            gaze_vector, 
                            last_gaze_vector
                        ) > FLIP_ANGLE_THRESHOLD:
                        # flip the gaze direction
                        gazing_point_2d[1] = gazing_point_2d[0] - gaze_vector
                        color = RED_COLOR

                    # Update the head pose history
                    old_head_pose = head_pose_history[person_id].pop(0)
                    head_pose_history[person_id].append(gazing_point_2d)
                    head_pose_estimator.draw_gazing_line(img, old_head_pose, color=CYAN_COLOR)

                # head_pose_estimator.draw_annotation_box(frame, 
                #     person_head_pose[0], 
                #     person_head_pose[1], color=CYAN_COLOR)

                # Draw the person gazing direction
                head_pose_estimator.draw_gazing_line(img, gazing_point_2d, color=color)

            for person_id in pose_data["persons"].keys():
                person = pose_data["persons"][person_id]
                    
                for body_part_id in person.keys():
                    kp = person[body_part_id]
                    kp[0] *= img_w 
                    kp[1] *= img_h
                    if kp[2] > 0.3:
                        img = cv2.circle(img, (int(kp[0]), int(kp[1])), 1, WHITE_COLOR, -1)

                for index, (start_joint_id, end_joint_id) in enumerate(getSkeleton()):
                    start_joint_id, end_joint_id = str(start_joint_id), str(end_joint_id)
                    start_joint_coor = person[start_joint_id]
                    end_joint_coor = person[end_joint_id]
                    color = getColorFromIndexSkeleton(index)
                    thickness = 1
                    if start_joint_coor[2] > 0.3 and end_joint_coor[2] > 0.3: 
                        img = cv2.line(img, (int(start_joint_coor[0]), int(start_joint_coor[1])), (int(end_joint_coor[0]), int(end_joint_coor[1])), color, thickness)

            if approach_data is not None:
                # Visualize box, id information
                if "Datetime" in approach_data.keys():
                    date_time = approach_data["Datetime"]
                    img = cv2.putText(img, "{}".format(date_time)\
                            , (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, \
                            WHITE_COLOR , 2, cv2.LINE_AA)
                    
                for person_id in approach_data.keys():
                    if person_id == "Datetime":
                                    continue
                    human = approach_data[person_id]
                    tl_ = human["tl_coord2d"]
                    tl = tl_ #(tl_.x, tl_.y)
                    br_ = human["br_coord2d"]
                    br = br_#(br_.x, br_.y)

                    if args.show_bounding_box: 
                        cv2.rectangle(img, (tl[0], tl[1]), \
                                    (br[0], br[1]), BLUE_COLOR, 1)
                        
                    if args.show_id:
                        if faction is not None:
                            action = action_data["actions"]
                            associative_action = action_data["associative_actions"]
                            if person_id in action and action[person_id]:
                                img = cv2.putText(img, "{}-{}".format(person_id, action[person_id][0]), \
                                    (tl[0], tl[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                                    WHITE_COLOR , 1, cv2.LINE_AA)
                        else:
                            img = cv2.putText(img, "{}".format(person_id),  \
                                (tl[0], tl[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                                WHITE_COLOR , 1, cv2.LINE_AA)
                    else:
                        if faction is not None:
                            action = action_data["actions"]
                            associative_action = action_data["associative_actions"]
                            if person_id in action and action[person_id]:
                                img = cv2.putText(img, "{}".format(action[person_id][0]), \
                                    (tl[0], tl[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                                    WHITE_COLOR , 1, cv2.LINE_AA)
                    
                    if faction is not None:
                        action = action_data["actions"]
                        associative_action = action_data["associative_actions"]   
                        if person_id in associative_action and associative_action[person_id]:
                            associative_act = associative_action[person_id][0].split("->")
                            act = associative_act[0]
                            second_person_id = associative_act[1]
                            if person_id in approach_data and second_person_id in approach_data:
                                # print("CONNECT", person_id, second_person_id, approach_data[person_id], approach_data[second_person_id])
                                img = drawline(img, approach_data[person_id], approach_data[second_person_id], act)

                    if args.student_staff_path:
                        if person_id in student_staff_data:
                            img = cv2.putText(img, "{}".format(student_staff_data[person_id]), \
                                            (tl[0], tl[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                                            WHITE_COLOR , 1, cv2.LINE_AA)
            out.write(img)
            print(idx)

    out.release()
    print("The video was successfully saved")

import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process visualizing pose logging')

    parser.add_argument("--pose_path", type=str, help="Path to .jl pose")
    parser.add_argument("--student_staff_path", type=str, default=None, help="Path to .jl student staff path")
    parser.add_argument("--action_path", type=str, default=None, help="Path to .jl action detection path")
    parser.add_argument("--background_path", type=str, default=None, help="Output video path")
    parser.add_argument("--output_path", type=str, default="./viz_pose.avi", help="save path")
    parser.add_argument("--show_id", action="store_true", help="Whether to show id of human")
    parser.add_argument("--show_bounding_box", action="store_true", help="Whether to show bouding box of human")
    parser.add_argument("--background_opacity", type=float, default=0.3, help="Background opacity")
    
    args = parser.parse_args()

    visualize(args)

