import json

# IOU
def calculate_iou(bbox1, bbox2):
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

def update_with_iou(pose_data, iou_threshold=0.6):
    # Make a copy of the input data to avoid modifying the original data
    updated_data = pose_data.copy()

    # Iterate over frames in pose_data
    for i in range(len(updated_data) - 1):
        frame1 = updated_data[i]
        frame2 = updated_data[i + 1]

        approach_data1 = frame1.get('approach', {})
        approach_data2 = frame2.get('approach', {})

        # Remove 'Datetime' key from approach_data
        del approach_data1['Datetime']
        del approach_data2['Datetime']

        # Get the person IDs in frame1 and frame2
        person_ids1 = list(approach_data1.keys())
        person_ids2 = list(approach_data2.keys())

        # Iterate over different ID pairs
        for person_id1 in person_ids1:
            for person_id2 in person_ids2:
                if person_id1 == person_id2: 
                    continue

                # Get the bounding boxes of the corresponding IDs in frame1 and frame2
                bbox1 = approach_data1.get(person_id1, {})
                bbox2 = approach_data2.get(person_id2, {})

                # Calculate the IoU between the two bounding boxes
                iou = calculate_iou(bbox1, bbox2)

                # If IoU exceeds the threshold, update the ID of person_id2 in frame2 to person_id1 in frame1
                if iou > iou_threshold:
                    approach_data2[person_id2]['id'] = approach_data1[person_id1]['id']

        # Update the approach data in frame2 with the updated IDs
        frame2['approach'] = approach_data2

    return updated_data

## EXTRACT
def extract_data(file_path) -> dict:
    data_dict = {}
    with open(file_path) as fin:
        # for each line in the json line file, each line corresponds to one frame
        for idx, line in enumerate(fin):
            # load the json line as a dictionary
            data = json.loads(line)
            data_dict[idx] = data
    return data_dict

def extract_bounding_boxes(line_data):
    """
    Extract bounding boxes from line_data
    Args:
        line_data: Dictionary containing the line data
    Returns:
        List of tuples, each tuple containing (tl_coord2d, br_coord2d) for each ID
    """
    bounding_boxes = []
    
    # Check if 'approach' key exists in line_data
    if 'approach' in line_data:
        approach_data = line_data['approach']
        
        # Iterate over IDs in approach_data
        for person_id, bbox_data in approach_data.items():
            if 'tl_coord2d' in bbox_data and 'br_coord2d' in bbox_data:
                tl_coord2d = bbox_data['tl_coord2d']
                br_coord2d = bbox_data['br_coord2d']
                bounding_boxes.append((tl_coord2d, br_coord2d))
    
    return bounding_boxes

# SAVE JL FILE
def save_file(data: dict, output_path):
    with open(output_path, 'w') as file:
        for idx in range(len(data)):
            file.write(json.dumps(data[idx]) + '\n')

