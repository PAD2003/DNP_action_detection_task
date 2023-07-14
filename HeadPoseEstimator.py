import cv2
import numpy as np

def scale_vector(v):
    max_abs = np.abs(v).max()
    if max_abs == 0:
        return v
    scale_factor = -np.log10(max_abs) + 2
    return v * 10 ** scale_factor

class HeadPoseEstimator():
    def __init__(self, img_size):
        self.focal_length = img_size[1]

        self.camera_center = (img_size[1] / 2, img_size[0] / 2)

        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array([[-14.97821226], [-10.62040383], [-2053.03596872]])
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")
        
        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))
        
        self.model_points_68 = self._get_full_model_points()
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Mouth left corner
            (150.0, -150.0, -125.0)      # Mouth right corner
        ]) / 4.5

    def draw_annotation_box(self, image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""
        point_3d = []
        rear_size = 7
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)
        

    def estimate_gazing_line(self, rotation_vector, translation_vector):
        """Draw a 3D box as annotation of pose"""
        point_3d = [(0,0,0), (0, 0, 500)]
        point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)
        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        # point_2d = np.int32(point_2d.reshape(-1, 2))
        point_2d = (point_2d.reshape(-1, 2))

        return point_2d

    def solve_pose_and_estimate_gazing_line_by_68_points(self, image_points):
        (rotation_vector, translation_vector) = self.solve_pose_by_68_points(image_points)
        gazing_point_2d = self.estimate_gazing_line(
            rotation_vector, 
            translation_vector
        )
        return gazing_point_2d

    def draw_gazing_line(self, image, point_2d, length=500, color=(255, 255, 255), line_width=1):
        gaze_vector = point_2d[1] - point_2d[0]
        point_2d[1] = point_2d[0] + scale_vector(gaze_vector)

        # Draw all the lines
        cv2.line(image, tuple(point_2d[0].astype(int)), tuple(point_2d[1].astype(int)), color, line_width, cv2.LINE_AA)

    def draw_axes(self, img, R, t):
        img	= cv2.drawFrameAxes(
            img, 
            self.camera_matrix, 
            self.dist_coeefs, R, t, 1300
        )

    def solve_pose(self, image_points):
        """
        Solve pose from image points
        Return (rotation_vector, translation_vector) as pose.
        """
        # import IPython ; IPython.embed()
        # assert image_points.shape[0] == self.model_points_68.shape[0], "3D points and 2D points should be of same number."
        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coeefs)

        return (rotation_vector, translation_vector)

    def solve_pose_by_68_points(self, image_points):
        """
        Solve pose from all the 68 image points
        image points are coordinates on the image (not normalized)
        Return (rotation_vector, translation_vector) as pose.
        """
        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_68, image_points, self.camera_matrix, self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_68,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)
        return (rotation_vector, translation_vector)

    def _get_full_model_points(self, filename='assets/head_pose_model.txt'):
        """Get all 68 3D model points from file"""
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T

        # Transform the model into a front view.
        model_points[:, 2] *= -1

        return model_points

    @staticmethod
    def draw_marks(image, marks, color=(255, 0, 0)):
        """Draw mark points on image"""
        for mark in marks:
            cv2.circle(image, (int(mark[0]), int(
                mark[1])), 1, color, -1, cv2.LINE_AA)