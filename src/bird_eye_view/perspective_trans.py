import cv2
import json
import numpy as np

# 4 points of floor
tl = (0.8274626865671642, 0.5513227513227513)
bl = (0.14567164179104478, 0.2571428571428571)
tr = (0.6053731343283583, 1.7883597883597884)
br = (-0.05791044776119403, 0.4264550264550265)

# original
original_width = 1280
original_height = 720

# output
output_width = 720
output_height = 1440

# functions
def extractData(file_path) -> dict:
    data_dict = {}
    with open(file_path) as file:
        for idx, line in enumerate(file):
            data = json.loads(line)
            data_dict[idx] = data
    return data_dict

def centerBoundingBox(bbox) -> tuple:
    [tl_coord2d_x, tl_coord2d_y] = bbox["tl_coord2d"]
    [br_coord2d_x, br_coord2d_y] = bbox["br_coord2d"]
    
    center_x = (tl_coord2d_x + br_coord2d_x) / 2
    center_y = (tl_coord2d_y + br_coord2d_y) / 2

    return (center_x, (center_y + br_coord2d_y) / 2)

def getAllCenters(frame_data):
    bboxs = frame_data["approach"]
    del bboxs["Datetime"]

    centers = {}
    for id, bbox in bboxs.items():
        centers[id] = centerBoundingBox(bbox)
    
    return centers

def getMiddleLegsPoint(frame_data):
    persons = frame_data["pose"]["persons"]

    points = {}
    for id, person_data in persons.items():
        point_x = (person_data["11"][0] + person_data["12"][0]) / 2
        point_y = (person_data["11"][1] + person_data["12"][1]) / 2

        # print((point_x, point_y))

        points[id] = (point_x * original_width, point_y * original_height)

    return points

# caculate matrix for perspective transformation
tl = (tl[0] * original_width, tl[1] * original_height)
bl = (bl[0] * original_width, bl[1] * original_height)
tr = (tr[0] * original_width, tr[1] * original_height)
br = (br[0] * original_width, br[1] * original_height)

pts1 = np.float32([tl, bl, tr, br])
pts2 = np.float32([[0,0], [0, output_height], [output_width, 0], [output_width, output_height]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)

# get transformed centers of all bounding box
transformed_centers = {}

data = extractData("output/re_id/pose_reid.jl")
for frame, frame_data in data.items():
    centers = getMiddleLegsPoint(frame_data)
    transformed_centers_one_frame = {}

    for label, (x, y) in centers.items():
        point_to_transform = np.array([[[x, y]]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(point_to_transform, matrix)
        transformed_centers_one_frame[label] = tuple(transformed_point[0][0])

    transformed_centers[frame] = transformed_centers_one_frame

# create video
output_path = "output/bird_eye_view/pose_bev_middleLegs.avi"
fps = 15

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height), isColor=True)

black_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)

for frame_idx, transformed_centers_one_frame in transformed_centers.items():
    frame = black_frame.copy()

    for label, (x, y) in transformed_centers_one_frame.items():
        cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), -1)
        cv2.putText(frame, str(label), (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    out.write(frame)

out.release()
cv2.destroyAllWindows()