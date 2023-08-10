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
# each line
{
	'pose': {
            'persons': {
                        "1": {
                                "0": [0.07742140028211805, 0.39094871278438303, 0.39686208963394165],
                                "1": [0.07700771755642362, 0.3858006612156276, 0.53644198179245],
                                ...
                            },
                        "2": { ... },
                        ...
                        }
            },

	'approach': {
					'Datetime': '2019-11-01 00:00:00',
					"1": {
							"tl_coord2d": [533, 184], 
							"br_coord2d": [647, 375], 
							"is_staff": false
					    },
					"2": {...},
					...
				}
}
```