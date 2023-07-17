import json
from funcions import update_with_iou

# Config
input_path = "data/pose.jl"
output_path = "output/pose_res.jl"

# Read data from the 'pose.jl' file
with open(input_path, 'r') as file:
    pose_data = [json.loads(line) for line in file]

# Create a list to store the updated ID results
pose_res_data = []

# Iterate over frames in pose_data
for line_data in pose_data:
    updated_data = update_with_iou(line_data)  # Update the IDs
    pose_res_data.append(updated_data)

# Write the result data to the 'pose_res.jl' file
with open(output_path, 'w') as file:
    for line_data in pose_res_data:
        file.write(json.dumps(line_data) + '\n')