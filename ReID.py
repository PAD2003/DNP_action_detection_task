import json
from functions import extract_data, save_file
from pose import dist
import cv2
import numpy as np
from visual_pose import  getColorFromIndexSkeleton, getSkeleton,WHITE_COLOR,BLUE_COLOR
import math

# BASIC
def extract_data(file_path) -> dict:
    data_dict = {}
    with open(file_path) as file:
        # for each line in the json line file, each line corresponds to one frame
        for idx, line in enumerate(file):
            # load the json line as a dictionary
            data = json.loads(line)
            data_dict[idx] = data
    return data_dict

def save_jl_file(data: dict, save_path) -> None:
    with open(save_path, 'w') as file:
        for key in data.keys():
            file.write(json.dumps(data[key]) + '\n')

def update_ids(persons: dict, bboxs: dict, map_ids: dict):
    persons_copy = persons.copy()
    for id in persons_copy.keys():
        if id in map_ids.keys():
            true_id = map_ids[id]

            persons[true_id] = persons.pop(id)
            bboxs[true_id] = bboxs.pop(id)

# VISUALIZE
def visualize_frame(data, frame_id, img_w, img_h):
    pose_data, approach_data = data[frame_id]["pose"], data[frame_id]["approach"]
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
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

                    cv2.rectangle(img, (tl[0], tl[1]), \
                                    (br[0], br[1]), BLUE_COLOR, 1)
                        
    
                    img = cv2.putText(img, "{}".format(person_id),  \
                                (tl[0], tl[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                                WHITE_COLOR , 1, cv2.LINE_AA)
    # cv2.imwrite("frame.jpg",img)
    return img

def visualize_vid(data,save_path, img_w, img_h):
    out = cv2.VideoWriter(
                    save_path,
                    cv2.VideoWriter_fourcc(*'MJPG'), 
                    15, (img_w, img_h))
    
    for frame_id in data.keys() : 
        img = visualize_frame(data,  frame_id, img_w, img_h)
        out.write(img)
    out.release()
    print("The video was successfully saved")

# KEYPOINTS
def dist_keypoints(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def dist_people(person_1: dict, person_2: dict) -> float:
    dist = 0
    for i in range(133):
        dist += dist_keypoints(person_1[str(i)], person_2[str(i)])
    
    return dist 

def confidence_keypoints(person_1: dict, person_2: dict) -> True:
    """
    return:
        True: person_1 more confidence
        False: person_2 more confidence
    """
    confidence_1 = 0
    confidence_2 = 0
    for key in person_1.keys():
        confidence_1 += person_1[key][2]
        confidence_2 += person_2[key][2]

    return confidence_1 > confidence_2

# BBOX
def iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) score between two bounding boxes
    Args:
        bbox1: Dictionary representing the first bounding box
        bbox2: Dictionary representing the second bounding box
    Returns:
        The IoU score
    """

    # Extract the coordinates of bbox1
    tl_coord1 = bbox1["tl_coord2d"]
    br_coord1 = bbox1["br_coord2d"]
    
    # Extract the coordinates of bbox2
    tl_coord2 = bbox2["tl_coord2d"]
    br_coord2 = bbox2["br_coord2d"]
    
    # Calculate the intersection coordinates
    x_left = max(tl_coord1[0], tl_coord2[0])
    y_top = max(tl_coord1[1], tl_coord2[1])
    x_right = min(br_coord1[0], br_coord2[0])
    y_bottom = min(br_coord1[1], br_coord2[1])
    
    # Calculate the intersection area
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    
    # Calculate the areas of bbox1 and bbox2
    bbox1_area = (br_coord1[0] - tl_coord1[0]) * (br_coord1[1] - tl_coord1[1])
    bbox2_area = (br_coord2[0] - tl_coord2[0]) * (br_coord2[1] - tl_coord2[1])
    
    # Calculate the union area
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Calculate the IoU score
    iou = intersection_area / union_area
    
    return iou

def area(bbox):
    tl_coord = bbox["tl_coord2d"]
    br_coord = bbox["br_coord2d"]

    width = br_coord[0] - tl_coord[0]
    height = br_coord[1] - tl_coord[1]

    return width * height

def intersection_area(bbox1, bbox2):
    tl1 = bbox1["tl_coord2d"]
    br1 = bbox1["br_coord2d"]
    tl2 = bbox2["tl_coord2d"]
    br2 = bbox2["br_coord2d"]

    x_left = max(tl1[0], tl2[0])
    y_top = max(tl1[1], tl2[1])
    x_right = min(br1[0], br2[0])
    y_bottom = min(br1[1], br2[1])

    width = max(0, x_right - x_left)
    height = max(0, y_bottom - y_top)

    intersection_area = width * height

    return intersection_area

# ReID
def remove_person(data: dict, area_threshold = 0.4, intersection_area_threshold = 0.6):
    removed_id_count = 0

    for frame_idx in data.keys():
        remove_id = []

        persons: dict = data[frame_idx]["pose"]["persons"]
        bboxs: dict = data[frame_idx]["approach"]

        for id_1 in bboxs:
            for id_2 in bboxs:
                if id_1 != id_2 and id_1 != "Datetime" and id_2 != "Datetime":
                    bbox1 = bboxs[id_1]
                    bbox2 = bboxs[id_2]

                    area1 = area(bbox1)
                    area2 = area(bbox2)
                    intersection = intersection_area(bbox1, bbox2)

                    if area1 < area2 * area_threshold and intersection > intersection_area_threshold * area1:
                        remove_id.append(id_1)
        
        for id in remove_id:
            if id in persons.keys():
                removed_id_count += 1
                persons.pop(id)
                bboxs.pop(id)
    
    return removed_id_count, data

def should_remove(bbox1, bbox2, area_threshold = 0.4, intersection_area_threshold = 0.6):
    area1 = area(bbox1)
    area2 = area(bbox2)
    intersection = intersection_area(bbox1, bbox2)

    if area1 < area2 * area_threshold and intersection > intersection_area_threshold * area1:
        return True
    
    return False

def dist_people_reid(data: dict, threshold = 10) -> dict:
    tracked_people = {} # {id (str) : person_data (dict)} # {"1" : {"0": [0.1, 0.1, 0.5], ... }, ... }
    map_ids = {} # {false_id (str) : true_id (str)}
    max_id = 0

    for frame_idx in data.keys():
        persons: dict = data[frame_idx]["pose"]["persons"]
        bboxs: dict = data[frame_idx]["approach"]

        update_ids(persons=persons, bboxs=bboxs, map_ids=map_ids)

        persons_copy = persons.copy()
        for id in persons_copy.keys():
            if int(id) <= max_id:
                tracked_people[id] = persons[id]
            else:
                flag = False
                for tracked_id in tracked_people.keys():
                    if dist_people(persons[id], tracked_people[tracked_id]) < threshold:
                        if tracked_id in persons_copy.keys():
                            persons.pop(id)
                            bboxs.pop(id)
                        else:
                            persons[tracked_id] = persons.pop(id)
                            bboxs[tracked_id] = bboxs.pop(id)
                            map_ids[id] = tracked_id
                        flag = True
                        break
                
                if flag == False:
                    max_id += 1
                    tracked_people[str(max_id)] = persons[id]
                    persons[str(max_id)] = persons.pop(id)
                    bboxs[str(max_id)] = bboxs.pop(id)

    print(map_ids)
    return max_id, data

def reid_2(data: dict, dist_threshold = 10, iou_threshold = 0.6) -> dict:
    tracked_people = {} # {id (str) : person_data (dict)} # {"1" : {"0": [0.1, 0.1, 0.5], ... }, ... }
    tracked_bboxs = {}
    map_ids = {} # {false_id (str) : true_id (str)}
    max_id = 0

    for frame_idx in data.keys():
        persons: dict = data[frame_idx]["pose"]["persons"]
        bboxs: dict = data[frame_idx]["approach"]

        update_ids(persons=persons, bboxs=bboxs, map_ids=map_ids)

        persons_copy = persons.copy()
        for id in persons_copy.keys():
            if int(id) <= max_id:
                tracked_people[id] = persons[id]
            else:
                flag = False
                for tracked_id in tracked_people.keys():
                    if dist_people(persons[id], tracked_people[tracked_id]) < dist_threshold:
                        if tracked_id in persons_copy.keys():
                            if should_remove(bboxs[tracked_id], bboxs[id]) == True:
                                persons.pop(tracked_id)
                                bboxs.pop(tracked_id)
                            else:
                                persons.pop(id)
                                bboxs.pop(id)
                        else:
                            persons[tracked_id] = persons.pop(id)
                            bboxs[tracked_id] = bboxs.pop(id)
                            map_ids[id] = tracked_id
                        flag = True
                        break
                
                if flag == False:
                    max_id += 1
                    tracked_people[str(max_id)] = persons[id]
                    tracked_bboxs[str(max_id)] = bboxs[id]
                    persons[str(max_id)] = persons.pop(id)
                    bboxs[str(max_id)] = bboxs.pop(id)

    return max_id, data

if __name__ == "__main__":
    # config
    file_path = "data/pose.jl"
    jl_file_save_path = "output/pose_reid_test.jl"

    # ReID
    data = extract_data(file_path=file_path)

    # removed_id_count, rm_data = remove_person(data)
    # print(f"Removed ID: {removed_id_count}")

    max_id, reid_data = dist_people_reid(data)
    print(f"Max ID: {max_id}")

    save_jl_file(data=reid_data, save_path=jl_file_save_path)

    # Visualize
    video_save_path = "output/pose_reid_test.avi"
    img_w = 1280
    img_h = 720
    visualize_vid(data=reid_data, save_path=video_save_path, img_w=img_w, img_h=img_h)


