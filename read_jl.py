import json
from pose import Pose

def export_pose_reid_file(args):
    with open(args.pose_path) as fin, open(args.output_path, "w") as fout:
        # for each line in the json line file, each line corresponds to one frame
        for idx, line in enumerate(fin):
            # load the json line as a dictionary
            data = json.loads(line)
            img_w, img_h = data["img_w"], data["img_h"]
            data_approach = data["approach"]
            data_pose = data["pose"]["persons"]

            for person_id in data_pose:
                person = data_pose[person_id]
                person_pose_obj = Pose(person)
                # head_coord = person['0']
            log(fout, data)

def log(fout, data):
    json.dump(data, fout)
    fout.write("\n")
