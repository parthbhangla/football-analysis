# Importing Libraries
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import sys
sys.path.append('../')
from utilities import get_centre_of_bbox, get_width_of_bbox

class Tracker:
    # Initializing the model
    def __init__(self, model_path):

        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    # Predicting for each frame
    def detect_frames(self, frames):

        batch_size = 20 # For memory efficiency
        detections = [] # Prediction of each frame
        for i in range (0,len(frames),batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections = detections + detection_batch
        return detections # Prediction of all frames in a list

    # Actual Tracking
    def track(self, frames, read_from_stub = False, stub_path = None):

        # Reading the tracks from the stub file if it exits
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)
        
        # Tracks for each objects
        tracks ={
            'players':[], # Of Form - [{0:{'bbox':[0,0,0,0]}},1:{'bbox':[0,0,0,0]}}] for each frame
            'referees':[],
            'ball':[]
        }

        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            class_names_inverse = {v: k for k, v in class_names.items()}
            
            # Converting detections to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Converting goalkeeper to player object for simplicity
            for object_index, class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_index] = class_names_inverse['player']
            
            # Tracking Objects - Adding a tracker to every detection
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # Appending the tracks to the tracks dictionary for each iteration
            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            # Adding the tracks to the dictionary with track_id for every frame - player and referee
            for frame_detection in detection_with_tracks:
                bounding_box = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == class_names_inverse['player']:
                    tracks['players'][frame_num][track_id] = {'bbox':bounding_box}

                if class_id == class_names_inverse['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox':bounding_box}

            # Adding the tracks to the dictionary with hard coded track_id for every frame - ball
            for frame_detection in detection_supervision:
                bounding_box = frame_detection[0].tolist()
                class_id = frame_detection[3]
    
                if class_id == class_names_inverse['ball']:
                    tracks['ball'][frame_num][1] = {'bbox':bounding_box}

        # Saving the tracks to a file
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    # Drawing the ellipse on the frame
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_centre_of_bbox(bbox)
        width = get_width_of_bbox(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        return frame # Frame with the ellipse drawn


    # Drawing the annotations on the frames
    def drawing_annotations(self, video_frames, tracks):
        
        output_video_frames =[]
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num] # Player Dictionary for the frame number
            ball_dict = tracks['ball'][frame_num] # Ball Dictionary for the frame number
            referee_dict = tracks['referees'][frame_num] # Referee Dictionary for the frame number

            # Drawing the annotations for players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player['bbox'], (0,0,255), track_id)
            
            output_video_frames.append(frame)

        return output_video_frames # Frames with the annotations drawn