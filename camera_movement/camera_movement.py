# Importing Libraries
import pickle
import cv2
import numpy as np
import sys
import os
sys.path.append('..')
from utilities import measure_xy_distance, measure_distance

class CameraMovementEstimator:

    def __init__(self, frame):
        self.minimum_distance = 5 # Minimum camera movement to be considered

        # Parameters for Lucas-Kanade Optical Flows
        self.lkparams = dict(
            winSize = (15, 15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Choosing first frame
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Taking only the upper half and lower half (they have the least movement)
        # Which makes it easier to track camera movement
        mask_feature = np.zeros_like(first_frame_grayscale)
        mask_feature[:, 0:20] = 1
        mask_feature[:, 900:1050] = 1

        # Selecting features to track
        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_feature
        )

    # Adjusting the tracking positions with the camera movement
    def adjust_tracking_positions(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    # Getting the camera movement
    def get_camera_movement(self, frames, read_from_stub = False, stub_path = None):

        # Reading from stub for efficiency
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0,0]]*len(frames) # Movement of x and y for each frame
        
        # In grayscale for better performance
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Enumerating from 1, since 0 is used already (as previous for the 1st frame)
        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lkparams)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            
            old_gray = frame_gray.copy()

        # Writing to stub for efficiency
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement
    
    # Drawing the camera movement
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 72), (255, 255, 255), -1)
            alpha = 0.6 # Transparency factor
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f'Camera Movement X: {x_movement:.2f}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 3)
            frame = cv2.putText(frame, f'Camera Movement Y: {y_movement:.2f}', (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 3)

            output_frames.append(frame)

        return output_frames