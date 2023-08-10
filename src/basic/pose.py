from typing import Dict
import math
import numpy as np

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    # return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** (1/2)

def limit_cos(value):
    if value > 1: value = 1
    if value < -1: value = -1
    return value

def vector_magnitude(v):
    return np.sqrt(np.sum(np.square(v)))

def polygon_area(points):
    """Compute the area of a 2D polygon given its input coordinates as a list of 2D numpy arrays."""
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[i][1] * points[j][0]
    area *= 0.5
    return abs(area)

"""
Calculate angle between 2 vectors
using the formular
cos(theta) = np.dot(u,v) / (|u|*|v|)
"""
def angle_vector(u, v):
    nominator = np.dot(u, v)
    denominator = vector_magnitude(u)*vector_magnitude(v)
    cos_angle =  nominator / denominator if denominator != 0 else 0
    cos_angle = min(cos_angle, 1)
    cos_angle = max(cos_angle, -1)
    return math.acos(cos_angle)

"""
Calculate angle p2->p1->p3 (p1 at middle)
"""
def angle(p1, p2, p3):
    a = math.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2)
    b = math.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
    c = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    nominator = (b**2 + c**2 - a**2)
    denominator = 2 * b * c

    if denominator == 0:
        return 0

    try:
        cos_angle =  nominator / denominator
        cos_angle = limit_cos(cos_angle)
        degree = math.degrees(math.acos(cos_angle))
    except Exception as e:
        print(e)
    return degree

class Pose():
    def __init__(self, pose_obj: Dict):
        """
        pose_obj: contains keypoints of a person
        Format: 
            {
                '0': [0.5, 0.1, 0.9],
                '1': [0.1, 0.2, 0.95],
                ...
                '132': [0.1, 0.2, 0.3]
            }
        """
        self.pose_obj = pose_obj

    def get_68_keypoints_for_head_pose_detection(self, scale_width=1, scale_height=1):
        person_marks = []
        for keypoint_id in range(23, 91):
            person_marks.append(self.pose_obj[str(keypoint_id)][:-1])
        person_marks = np.array(person_marks).astype("float32")
        person_marks[:,0] *= scale_width
        person_marks[:,1] *= scale_height
        return person_marks

    def left_hand_coord(self):
        return self.pose_obj['9'][0], self.pose_obj['9'][1]

    def right_hand_coord(self):
        return self.pose_obj['10'][0], self.pose_obj['10'][1]

    def left_elbow_coord(self):
        return self.pose_obj['7'][0], self.pose_obj['7'][1]

    def right_elbow_coord(self):
        return self.pose_obj['8'][0], self.pose_obj['8'][1]

    def left_shoulder_coord(self):
        return self.pose_obj['5'][0], self.pose_obj['5'][1]

    def right_shoulder_coord(self):
        return self.pose_obj['6'][0], self.pose_obj['6'][1]
    
    def left_ear_coord(self):
        return self.pose_obj['3'][0], self.pose_obj['3'][1]

    def right_ear_coord(self):
        return self.pose_obj['4'][0], self.pose_obj['4'][1]
    
    def left_eye_coord(self):
        return self.pose_obj['1'][0], self.pose_obj['1'][1]

    def right_eye_coord(self):
        return self.pose_obj['2'][0], self.pose_obj['2'][1]
    
    def head_coord(self):
        return self.pose_obj['0'][0], self.pose_obj['0'][1]

    def left_leg_coord(self):
        return self.pose_obj['11'][0], self.pose_obj['11'][1]

    def right_leg_coord(self):
        return self.pose_obj['12'][0], self.pose_obj['12'][1]

    def left_knee_coord(self):
        return self.pose_obj['13'][0], self.pose_obj['13'][1]

    def right_knee_coord(self):
        return self.pose_obj['14'][0], self.pose_obj['14'][1]

    def left_ankle_coord(self):
        return self.pose_obj['15'][0], self.pose_obj['15'][1]
    
    def right_ankle_coord(self):
        return self.pose_obj['16'][0], self.pose_obj['16'][1]
    
    def hand_dist(self):
        return dist(
            self.left_hand_coord(),
            self.right_hand_coord(),
        )

    def head_left_hand_dist(self):
        return dist(
            self.head_coord(),
            self.left_hand_coord()
        )
    
    def head_right_hand_dist(self):
        return dist(
            self.head_coord(),
            self.right_hand_coord()
        )

    def shoulder_dist(self):
        return dist(
            self.right_shoulder_coord(),
            self.left_shoulder_coord()
        )

    def left_elbow_angle(self):
        return angle(
            self.left_elbow_coord(),
            self.left_shoulder_coord(),
            self.left_hand_coord()
        )

    def right_elbow_angle(self):
        return angle(
            self.right_elbow_coord(),
            self.right_shoulder_coord(),
            self.right_hand_coord()
        )

    def left_shoulder_angle(self):
        return angle(
            self.left_shoulder_coord(),
            self.left_elbow_coord(),
            self.left_leg_coord()
        )

    def right_shoulder_angle(self):
        return angle(
            self.right_shoulder_coord(),
            self.right_elbow_coord(),
            self.right_leg_coord()
        )

    def left_leg_angle(self):
        return angle(
            self.left_knee_coord(),
            self.left_leg_coord(),
            self.left_ankle_coord()
        )
    
    def right_leg_angle(self):
        return angle(
            self.right_knee_coord(),
            self.right_leg_coord(),
            self.right_ankle_coord()
        )
    
    def left_hip_angle(self):
        return angle(
            self.left_leg_coord(),
            self.left_shoulder_coord(),
            self.left_knee_coord()
        )
    
    def right_hip_angle(self):
        return angle(
            self.right_leg_coord(),
            self.right_shoulder_coord(),
            self.right_knee_coord(),
        )

    def average_coords(self):
        np_pose_keypoints = np.array(list(self.pose_obj.values()))[:,:-1]
        person_coord_x, person_coord_y = np.average(np_pose_keypoints, axis=0)

        return (person_coord_x, person_coord_y)

    """
    Return list of points which form the boundary of the face
    """
    def face_boundary(self):
        return [
            self.head_coord(),
            self.right_ear_coord(),
            self.right_eye_coord(),
            self.left_eye_coord(),
            self.left_ear_coord()
        ]

    def face_area(self):
        return polygon_area(self.face_boundary())