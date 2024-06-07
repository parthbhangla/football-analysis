# Importing Libraries
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import sys
from utilities import get_centre_of_bbox, get_width_of_bbox
import numpy as np
import pandas as pd
sys.path.append('../')

class Tracker:
    # Initializing the model
    def __init__(self, model_path):

        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    # Interpolating The Ball For Frames It Is Not Detected
    def interpolate_ball_position(self, ball_positions):

        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1','y1','x2','y2'])

        # Interpolating the missing ball positions
        df_ball_positions = df_ball_positions.interpolate()

        # Edge case where the first frame is missing by backfilling
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1 : {'bbox' : x}}for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

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
    
        # Drawing the rectangle on the frames
        rectangle_width = 40
        rectangle_height = 20

        # To centre the rectangle and give it some buffer
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2 + 15)
        y2_rect = (y2 + rectangle_height//2 + 15)

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect),int(y1_rect)),
                (int(x2_rect),int(y2_rect)),
                color,
                cv2.FILLED
                )

            # Drawing the text on the frames
            x1_text = x1_rect + 12

            # Spacing the larger track ids evenly on the rectangle
            if track_id > 99:
                x1_text = x1_text - 10
            cv2.putText(
                frame,
                f'{track_id}',
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame # Frame with the ellipse drawn
    
    # Drawing triangle for the ball on each frame to track it
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1]) # Triangle is on the top of the ball
        x , _ = get_centre_of_bbox(bbox) # Centre of the ball
        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)

        return frame # Frame with the triangle drawn

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
                color = player.get('team_color', (0,0,255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)

            # If player has the ball, draw a triangle on the player
            if player.get('has_ball', False):
                frame = self.draw_triangle(frame, player['bbox'], (0,0,255))

            # Drawing the annotations for referees
            for _ , referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0,255,255), track_id=None)

            # Drawing the annotations for the ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0,255,0))
            
            output_video_frames.append(frame)

        return output_video_frames # Frames with the annotations drawn