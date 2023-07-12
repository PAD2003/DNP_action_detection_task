# action_detection_task

## TASK 1: REID
### INPUT: pose.jl
### OUTPUT: pose_reid.jl

## TASK 2: STUDENT STAFF CLASSIFICATION
### INPUT pose_reid.jl
### OUTPUT: stsf.jl

## TASK 3: ACTION DETECTION
### INPUT: stsf.jl, pose_reid.jl
### OUTPUT: action.jl

```python
# pose.jl structure
line 1 = {
    'img_w': 1920,
    'img_h': 1080,
    'pose': 
        {
            'Datetime': '2019-11-01 00:00:00',
            'persons': 
            { # id
                "1":
                "2":
                "3": 
                { # pose keypoint
                    "0": [0.07742140028211805, 0.39094871278438303, 0.39686208963394165],
                    "1": [0.07700771755642362, 0.3858006612156276, 0.53644198179245],
                    "2":
                    "3"
                    ...
                }
            }
        },
    'approach':
    {
        { # id
                "1":
                "2":
                "3": 
                {
                    "tl_coord2d": [533, 184], 
                    "br_coord2d": [647, 375], 
                    "is_staff": false # skip
                }
    }
```