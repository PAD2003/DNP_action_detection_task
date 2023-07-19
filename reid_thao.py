import cv2 
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from collections import defaultdict

from HeadPoseEstimator import HeadPoseEstimator
from pose import angle_vector, Pose
from visual_pose import  getColorFromIndexSkeleton, getSkeleton,WHITE_COLOR,CYAN_COLOR,RED_COLOR,BLUE_COLOR,GREEN_COLOR, ORANGE_COLOR, PURPLE_COLOR, YELLOW_COLOR

import json
"""
data[i]["pose"]["persons"] : List persons in i-th frame
data[i]["approach"][]
"""
#Convert ij to dict 
def convert_jl_to_dict(pose_path):
    data_dict = {}
    index = 0
    with open(pose_path, 'r') as file:
        for line in file:
            json_data = json.loads(line.strip())
            data_dict[index] = json_data
            index += 1
    return data_dict


#Convert dict to jl 
def convert_dict_to_jl(dictionary, output_file):
    with open(output_file, 'w') as file:
        for key in dictionary.keys():
            file.write(json.dumps(dictionary[key]) + '\n')


#distance between 2 kpoints
def disc_2_kpoint(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

#distance between 2 people:
def disc_2_persons(p1,p2):
    disc = 0
    for i in range(133) :
        disc += disc_2_kpoint(p1[str(i)],p2[str(i)])
    
    return disc 
def update_id(persons,bboxs, map_id):
    """
    mapid = {false id : true id , }
    """
    p = persons.copy()
    for person_id in p.keys():
        if person_id in map_id :
            persons[map_id[person_id]] = persons.pop(person_id)
            bboxs[map_id[person_id]] = bboxs.pop(person_id)

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

def reID(data, threshold = 10):
    map_id = {}
    vid = {}
    max_id = 0
    #  total_frames = len(data.keys())
    for frame_id in data.keys():
        persons = data[frame_id]["pose"]["persons"]
        bboxs = data[frame_id]["approach"]
        if map_id:
            update_id(persons, bboxs,map_id)
        p = persons.copy()
        for person_id in p.keys():
            #check if vid none then push 
            if not vid:
                vid[person_id] = persons[person_id]
                max_id = max_id + 1
            else:
                if int(person_id) <= max_id : #Already in vid then update
                    vid[person_id] = persons[person_id]
                
                else : #check
                    new_id = True
                    for id in vid.keys():
                        if( disc_2_persons(vid[id], persons[person_id]) < threshold) : #same person
                            if(id in p.keys()): #already in 
                                persons.pop(person_id) #remove duplicate 
                                bboxs.pop(person_id)

                            else : 
                                persons[id] = persons.pop(person_id) #re ID
                                bboxs[id] = bboxs.pop(person_id)
                                map_id[person_id] = id
                            
                            new_id = False
                            break
                    if new_id : 
                        max_id = max_id + 1
                        persons[str(max_id)] = persons.pop(person_id) #Re ID
                        bboxs[str(max_id)] = bboxs.pop(person_id)
                        vid[str(max_id)] = persons[str(max_id)] # Update
    return max_id, data
        # print(max_id)


if __name__=="__main__":
    pose_path = "data/pose.jl"
    img_w = 1280
    img_h = 720
    save_path = "output/reId_vid.avi"
    data = convert_jl_to_dict(pose_path=pose_path)
    #  visualize_frame(data, 1, img_w, img_h)
    #  persons = data[1]["pose"]["persons"]
    #  print(disc_2_persons(persons["1"], persons["2"]))
    maxId, new_data = reID(data)
    # visualize_frame(new_data,112, img_w, img_h)
    # print(len(new_data.keys()))
    print(maxId)
    # reID(data)
    visualize_vid(new_data, save_path, img_w, img_h)