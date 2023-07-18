import json
import math
from pose import *

# Read js file
def read_line_Nth(args):
    with open(args.pose_path) as f:
        for idx, line in enumerate(f):
            if idx == args.line_nth - 1:
                line_json_data = json.loads(line)
                break
    # print (line_json_data['approach'].keys())
    return line_json_data


# Write a line from dict  to file.js
def write_file(args, line_json_data = None):
    if line_json_data == None:
        line_json_data = read_line_Nth(args)
    with open(args.output_path, "w") as fout:
        log(fout, line_json_data)

def log(fout, data):
    json.dump(data, fout)
    fout.write("\n")

#* Print key of a dict
def Print_keys(Dict):
    print(Dict.keys())


#! Change id of a frame
def change_id(line_json_data_default, old_id= None ,new_id=None):
    '''
    a line of .jl file will like
    {'pose': {
        'person': {
            id': i.e '12' {
                'keypoint_0': [0.0234523456236, 0.021342154, 0.0215156], 
                'keypoint_1': ...
                'keypoint_2': ...
                
            }
        }, 'Datetime': ...,}
        'appoarch': {
            'Datetime': ....,
            'id': {
                'tl_coord2d': [87, 256], 'br_coord2d': [148, 330], 'is_staff': False
            }
            ...
            
        }
    'img_w': 1280,
    'img_h': 720
    }
    '''
    
    """
    We need to change on both persons and approach
        
    """
    
    line_json_data = line_json_data_default.copy()
    
    #Person
    person_data = line_json_data['pose']['persons']
    #* "id":{'keypoint_0': [0.0234523456236, 0.021342154, 0.0215156], }
    
    approach_data = line_json_data['approach']
    
    
    '''
    'Datetime': ....,
    'id': {
        'tl_coord2d': [87, 256], 'br_coord2d': [148, 330], 'is_staff': False
    }
    '''
    
    #! change id
    print("old_id = {}, new_id = {}".format(old_id, new_id ))
    
    line_json_data['pose']['persons'][new_id] = line_json_data['pose']['persons'].pop(old_id)
    line_json_data['approach'][new_id] = line_json_data['approach'].pop(old_id)
    
    return line_json_data



#denormanlize keypoint in each bbox 
def denormalize(keypoints, bbox):
    height = bbox['br_coord2d'][1] - bbox['tl_coord2d'][1]
    width = bbox['br_coord2d'][0] - bbox['tl_coord2d'][0]
    # keypoint[0] = width*keypoint[0] + bbox['tl_coord2d'][0]
    # keypoint[1] = height*keypoint[1] + bbox['tl_coord2d'][1]
    
    pose_obj = Pose(keypoints)
    person_mark = pose_obj.get_68_keypoints_for_head_pose_detection(scale_height= height, scale_width=width)
    
    return person_mark



# Compare with threshold to change the id
def calculate_2_frame_keypoint(ID1_keypoints,ID1_bbox ,ID2_keypoints, ID2_bbox):
    sum = 0
    
    person_mark_1 = denormalize(ID1_keypoints, ID1_bbox)
    person_mark_2 = denormalize(ID2_keypoints, ID2_bbox)
    
    for i in range(len(person_mark_1)):
        sum += dist(person_mark_1[i], person_mark_2[i])
    sum = sum / float(len(person_mark_1))
    return sum




def match_id(args):
    Frames = []
    with open(args.pose_path) as f:
        for idx, line in enumerate(f):
            Frames.append(json.loads(line)) #each line is a frame
    
    for i in range(len(Frames)-1):
        line_json_data_before = Frames[i]
        line_json_data_after = Frames[i+1]
        
        person_before = line_json_data_before['pose']['persons']
        approach_before = line_json_data_before['approach']

        person_after = line_json_data_after ['pose']['persons']
        approach_after  = line_json_data_after ['approach']

        
        id_olds = []
        id_news = []        
        print('======================Frame{}======================\n'.format(i) )
        print("**erson_before**")  
        print(person_before.keys())
        for id_before in list(person_before):
            
            for id_after in list(person_after):
                
                if False:
                    a = 1
                else :
                    diff = calculate_2_frame_keypoint(person_before[id_before], approach_before[id_before], person_after[id_after], approach_after[id_after])
                    
                    if diff < args.threshold:
                        print("id_bf: {} and id_af: {} diff = ".format(id_before, id_after) + str(diff))
                        print("diff < args.threshold")
                        Frames[i+1]=change_id(line_json_data_after, id_after, id_before)
        print("##person_after##")  
        print(person_after.keys())
    return Frames

def save_jl(args, Frames):
    with open(args.output_path, "w") as fout:
        for line in Frames:
            log(fout, line)



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process on file .js')
    parser.add_argument("--pose_path", type=str, help="path to .jl file")
    parser.add_argument("--output_path", type=str, help="new .jl file")
    parser.add_argument("--line_nth", type=int, help="the line nth you want to read")
    parser.add_argument("--threshold", type= float, help = "threshold to change id")
    
    
    args = parser.parse_args()
    
    
    
    Frames = match_id(args)
    save_jl(args, Frames)
    
    # line_json_data = read_line_Nth(args)
    # line_json_data = change_id(line_json_data, "1", "70")
    
    # person_data = line_json_data['pose']['persons']
    # #* "id":{'keypoint_0': [0.0234523456236, 0.021342154, 0.0215156], }
    
    # approach_data = line_json_data['approach']
    
    # Print_keys(person_data)
    # Print_keys(approach_data)
    
    # sum = calculate_2_frame_keypoint(person_data["12"], approach_data['12'], person_data["11"], approach_data['11'])
    
    # print(sum)
    # write_file(args, {"pose": person_data, "approach": approach_data})
