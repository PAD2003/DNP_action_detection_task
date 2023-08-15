import json
import numpy as np
from scipy.stats import norm

# BASIC
def extractData(file_path) -> dict:
    data_dict = {}
    with open(file_path) as file:
        # for each line in the json line file, each line corresponds to one frame
        for idx, line in enumerate(file):
            # load the json line as a dictionary
            data = json.loads(line)
            data_dict[idx] = data
    return data_dict

def saveSTSFFile(staff_ids, max_id, file_path) -> None:
    stsf = {}

    for i in range(1, max_id + 1):
        i_str = str(i)
        if i_str in staff_ids:
            stsf[i_str] = "staff"
        else:
            stsf[i_str] = "student"
    
    with open(file_path, 'w') as file:
        file.write(json.dumps(stsf))

# HELP FUNCTIONS
def findMaxId(data) -> int:
    pass

def personInFrame(data, id, frame) -> bool:
    id = str(id)
    return id in data[frame]["approach"]

def centerBoundingBox(bbox) -> tuple:
    [tl_coord2d_x, tl_coord2d_y] = bbox["tl_coord2d"]
    [br_coord2d_x, br_coord2d_y] = bbox["br_coord2d"]
    
    center_x = (tl_coord2d_x + br_coord2d_x) / 2
    center_y = (tl_coord2d_y + br_coord2d_y) / 2

    return (center_x, center_y)

def distBoundingBoxsSameIDIn2Frame(data, id, frame_1, frame_2) -> float:
    bbox_frame_1 = data[frame_1]["approach"][str(id)]
    bbox_frame_2 = data[frame_2]["approach"][str(id)]

    center_bbox_frame_1 = centerBoundingBox(bbox_frame_1)
    center_bbox_frame_2 = centerBoundingBox(bbox_frame_2)

    return np.linalg.norm(np.array(center_bbox_frame_1) - np.array(center_bbox_frame_2)) # EUCLID distance

# DISTANCE
def totalDistance(data, max_id, frame_gap=10) -> dict:
    total_frame = len(data)
    res = {}

    for i in range(1, max_id + 1):
        res[str(i)] = 0

        frame_curr = 0
        frame_begin = 0

        for tmp in range(total_frame):
            if personInFrame(data, i, tmp):
                frame_begin = tmp
                frame_curr = tmp
                break

        while frame_curr < total_frame:
            count = frame_gap

            while count != 0 and frame_curr < total_frame - 1:
                frame_curr += 1

                if personInFrame(data, i, frame_curr):
                    count -= 1
            
            if personInFrame(data, i, frame_curr):
                res[str(i)] += distBoundingBoxsSameIDIn2Frame(data, i, frame_begin, frame_curr)
            frame_begin = frame_curr

            frame_curr += 1
    
    return res

# NUMBER OF STAFF
def getAllCenters(frame_data):
    bboxs = frame_data["approach"]
    del bboxs["Datetime"]

    centers = {}
    for id, bbox in bboxs.items():
        centers[id] = centerBoundingBox(bbox)
    
    return centers

def frequencyAtEdge(data, max_id, prob_threshold) -> dict:
    """
    Returns:
        dict:
            key: id
            value: frequency stay at edge
    """
    res = {}
    for i in range(1, max_id + 1):
        res[str(i)] = 0
    
    for frame_data in data.values():
        centers = getAllCenters(frame_data)

        centers_array = np.array(list(centers.values()))

        mean_x = np.mean(centers_array[:, 0])
        mean_y = np.mean(centers_array[:, 1])

        std_x = np.std(centers_array[:, 0])
        std_y = np.std(centers_array[:, 1])

        for id, (center_x, center_y) in centers.items():
            prob = norm.pdf(center_x, loc=mean_x, scale=std_x) * norm.pdf(center_y, loc=mean_y, scale=std_y)
            if prob < prob_threshold:
                res[id] += 1

    return res

# INTERSECTION
def getIntersectionTopOf2Dict(dict_1, dict_2, top = 3):
    first_keys_dict_1 = list(dict_1.keys())[:top]
    first_keys_dict_2 = list(dict_2.keys())[:top]

    intersect_keys = set(first_keys_dict_1) & set(first_keys_dict_2)

    return intersect_keys

# RUN
if __name__ == "__main__":
    # config
    max_id = 19
    data_file_path = "output/re_id/pose_reid.jl"
    stsf_file_path = "output/staff_student/stsf.jl"

    # get data
    data = extractData(data_file_path)

    # total distance
    total_dist = totalDistance(data, max_id)
    sorted_total_dist = {k: v for k, v in sorted(total_dist.items(), key=lambda item: item[1], reverse=True)}
    print(sorted_total_dist)

    # frequency at edge
    frequency_at_edge = frequencyAtEdge(data, max_id, 0.000002)
    sorted_frequency_at_edge = {k: v for k, v in sorted(frequency_at_edge.items(), key=lambda item: item[1], reverse=True)}
    print(sorted_frequency_at_edge)

    # save result
    staff_ids = getIntersectionTopOf2Dict(sorted_total_dist, sorted_frequency_at_edge, 3)
    print(staff_ids)
    saveSTSFFile(staff_ids, max_id, stsf_file_path)
