import cv2 
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import json

from ..re_id.ReID import dist_keypoints

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

def list_id_keypoint(data, id_kp):
    """
    Args : 
        data
    Returns:
        person : Dict {frame_id : (x,y) of id_kp}
    Idea:
        lay duoc list id_kp qua tung frame
    """
    person = {}
    for frame_id in data.keys():
            # for frame_id in data.keys():
        persons = data[frame_id]["pose"]["persons"]
        bboxs = data[frame_id]["approach"] 
        for id in persons.keys():
            x, y = persons[id][str(id_kp)][0], persons[id][str(id_kp)][1]
            p = [x,y ]
            if id in person:
                person[id].extend(list([p]))
            else:
                person[id] = list([p])
    return person
    

def draw_bb(data):
    """
    Args:
        data: dict (approach (id, bbox))
    Returns:
        distance : dict ("id_person" : [list of distance])
        position : dict ("id_person" : [list of center_coordinate])
    Ideas:
        distance : Tinhs khoang cach chenh lech cua 1 nguoi qua 15 frame
        position : Tinh vi tri center cua bbox qua tung frame
    """
    bboxs = {}
    for frame_id in data.keys():
        bboxs[frame_id] = data[frame_id]["approach"]
        bboxs[frame_id].pop("Datetime")
        # print(data[frame_id]["approach"])
        for id in bboxs[frame_id].keys():
            # print(bboxs[frame_id].keys())
            # if id == "Datetime":
            tl = bboxs[frame_id][id]["tl_coord2d"]
            br = bboxs[frame_id][id]["br_coord2d"]
            center = [(tl[0] + br[0]) /2 , (tl[1] + br[1])/2]
            bboxs[frame_id][id]["center"] = center
    
    frame_id = 0
    flag = True
    length = len(data.keys())
    distance = {}
    position = {}
    while flag :
        prev = frame_id
        nex = frame_id  + 15
        if nex >= length : 
            nex = length -1
            flag = False
        for id in bboxs[nex].keys():
            if id in bboxs[prev].keys():
                p1 = bboxs[nex][id]["center"]
                p2 = bboxs[prev][id]["center"]
                # disc = max(abs(p1[0]-p2[0]), abs(p2[0]- p1[0])) #max of x,y
                disc = dist_keypoints(p1,p2)
                if id in position : 
                    position[id].extend(list([p2]))
                else :
                    position[id] = list([p1])
                if id in distance : 
                    distance[id].extend(list([disc]))
                else:
                    distance[id] = list([disc])
        frame_id = frame_id + 1


    return distance,position
def calculate_totaldistance(dis):
    """
    Args : 
        dis : dict = ("id_person" : [list of distances])
    Returns:
        total : dict = ("id_person" : total distance)
    Idea:
        tinh tong khoang canh di chuyen dc cua tung nguoi 
    """
    total = {}
    for id in dis.keys():
        total[id] = sum(dis[id])
    return total

def calculate_angle(p1, p2 ,p3):
    x_1 = p1[0] - p2[0]
    y_1 = p1[1] - p2[1]
    x_2 = p3[0] - p2[0]
    y_2 = p3[1] - p2[1]

    l1 = float(math.sqrt(x_1**2 + y_1**2 + 0.1 ))
    l2 = float(math.sqrt(x_2**2 + y_2**2 + 0.1))
    angle_radians = math.acos( x_1*x_2 + y_1*y_2/ (l1 * l2))

    # Convert the angle from radians to degrees
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees
    
def angle(person):
    p7 = person[str(7)]
    p13 = person[str(13)]
    p15 = person[str(15)]
    p6 = person[str(6)]
    p12 = person[str(12)]
    p14 = person[str(14)]
    ans1 = 0
    ans2 = 0
    if p7[2] > 0.3 and p13[2] > 0.3 and p15[2] > 0.3 :
        ans1 = calculate_angle(p7,p13, p15)
    if p6[2] > 0.3 and p12[2] > 0.3 and p14[2] > 0.3 : 
        ans2 =  calculate_angle(p6, p12, p14)
    ans = max(ans1, ans2)
    if ans == 0:
        return False
    return ans

def list_angle(data):
    """
    kp : 7-13-15
    kp : 6-12-14
    """
    ans = {}
    for frame_id in data.keys():
            # for frame_id in data.keys():
        persons = data[frame_id]["pose"]["persons"]
        bboxs = data[frame_id]["approach"] 
        for id in persons.keys():
            person = persons[id]
            if angle(person) is not False: 
                if id in ans :
                    ans[id].extend(list([angle(person)]))
                else :
                    ans[id] = list([angle(person)])
            
    return ans

def normalize(dis):
    """
    Chuan hoa khoang cach ve [0,1] -> Ve gaussian (khong can thiet lam)
    """
    dis_norm = {}
    for id in dis.keys():
        data= dis[id]
        data_min = np.min(data)
        data_max = np.max(data)
        data_normalized = (data - data_min) / (data_max - data_min)
        dis_norm[id] = data_normalized
    return dis_norm
def generate_gaussian(dis):
    # gaussian_param = {}
    x = np.linspace(-1, 1, 1000)
    for id in dis.keys():
        mu, sigma = norm.fit(dis[id])
        gaussian_curve = norm.pdf(x, mu, sigma)
        plt.plot(x, gaussian_curve, label=f'{2}')
    plt.xlabel('Values')
    plt.ylabel('Probability Density')
    plt.title('Multiple Gaussian Curves')
    plt.legend()
    plt.grid(True)
    

# Show the plot with all the Gaussian curves
    plt.show()

def visualize_distances(dis):

    for key, value in dis.items():
        # y = np.linspace(0, 200, 1000)
        # y = x ** 2
        fig, ax = plt.subplots()
        ax.plot(value, label=f'Frame {key}')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Euclidean Distance')
        ax.set_title('Distance between Measurement and Prediction')
        ax.legend()
        # plt.savefig(f'plot/id{key}.png')
        # plt.ylim(0, 200)  # Set the y-axis limits to [0, 200]
        plt.savefig(f'disc/id{key}.png')
        plt.close()  # Close the current figure to release memory
        

    # ax.set_xlabel('Frame')
    # ax.set_ylabel('Euclidean Distance')
    # ax.set_title('Distance between Measurement and Prediction')
    # ax.legend()
    # plt.show()
def visualize_pos(pos):
    # import matplotlib.pyplot as plt

# Sample list of points in the form of (x, y) coordinates
    for id in pos.keys():
        fig, ax = plt.subplots()
        point = pos[id]
        x, y = zip(*point)
        # x_values, y_values = zip(*points)

# Convert the x and y arrays into NumPy arrays
        x =np.array(x)
        y = np.array(y)
        # x_n = (x - np.min(x)) / (np.max(x) - np.min(x))
        # y_n = (y - np.min(y)) / (np.max(y) - np.min(y))




# Combine the normalized x and y coordinates back into (x, y) pairs
# normalized_points = list(zip(x_normalized, y_normalized))
        ax.plot(x, y, linestyle='-', color='b')
        
   

# Add labels and title
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Scatter Plot of Points')
        # plt.ylim(0, 500)
        # plt.xlim(0, 500)
        plt.savefig(f'pos/id{id}.png')
        

# Show the plot
    # plt.show()

if __name__ == "__main__":
    # config
    file_path = "output/pose_reid_test.jl"
    data = convert_jl_to_dict(pose_path=file_path)
    # bboxs = {}
    # bboxs[2] = data[2]["approach"]
    # print(bboxs[2]["2"].keys())
    dis, pos= draw_bb(data)
    # total = calculate_totaldistance(dis)
    # Sort the dictionary by values in descending order
    # sorted_dict = sorted(total.items(), key=lambda item: item[1], reverse=True)

    # Print the sorted dictionary (ranked by values in descending order)
    # for key, value in sorted_dict:
    #     print(f"{key}: {value}")

    # for id in total.keys():
    #     print(f"id {id} : total {total[id]}")
    # listangle = list_angle(data)
    # print(listangle)
    # visualize_distances(listangle)
    # # plt.plot(dis["2"])
    # # plt.xlabel('Frame')
    # # plt.ylabel('Euclidean Distance')
    # # plt.title('Distance between Measurement and Prediction')
    # # plt.show()
    # visualize_pos(pos)
    dis_norm = normalize(dis)
    # id_13 = list_id_keypoint(data,13)
    # print(id_13)
    # visualize_pos(id_13)
    generate_gaussian(dis_norm)
    # print(dis)


    