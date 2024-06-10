# Importing libraries
import sys
sys.path.append('../')
from utilities import measure_distance
import cv2
from utilities import get_foot_position

class SpeedAndDistanceEstimator():
    def __init__(self):

        # Calculating speed in windows of 5 frames
        self.frame_window = 5
        self.frame_rate = 24

    # Adding speed and distance to the track ids after the calculations
    def add_speed_and_distace_to_tracks(self, tracks):

        total_distance = {}

        for object, object_tracks in tracks.items():

            # Not calculating distance and speed for ball and referee
            if object == 'ball' or object == 'referee':
                continue
            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, _ in object_tracks[frame_num].items():

                    # If track id (aka the player) is not in the last frame, continue (makes it easier)
                    if track_id not in object_tracks[last_frame]:
                        continue

                    # Calculating distance and speed for each track id
                    start_position = object_tracks[frame_num][track_id]['position_transformed'] # Position in metres
                    end_position = object_tracks[last_frame][track_id]['position_transformed'] # End position in metres

                    # If player not in the field of view
                    if start_position == None or end_position == None:
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate # Time in seconds
                    speed_metres_per_second = distance_covered / time_elapsed
                    speed_kmph = speed_metres_per_second * 3.6

                    if object not in total_distance:
                        total_distance[object] = {}

                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in object_tracks[frame_num_batch]:
                            continue
                        object_tracks[frame_num_batch][track_id]['distance'] = total_distance[object][track_id]
                        object_tracks[frame_num_batch][track_id]['speed'] = speed_kmph

    # Drawing speed and distance on the frame
    def draw_speed_and_distance(self, frames, tracks):
        
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():

                # Not for the ball or the referee
                if object == "ball" or object == "referees":
                    continue 

                for _, track_info in object_tracks[frame_num].items():
                   if "speed" in track_info:
                       speed = track_info.get('speed',None)
                       distance = track_info.get('distance',None)
                       if speed is None or distance is None:
                           continue
                       
                       bbox = track_info['bbox']
                       position = get_foot_position(bbox)
                       position = list(position)
                       position[1]+=40 # Buffer for the text

                       position = tuple(map(int,position))
                       cv2.putText(frame, f"{speed:.2f} km/h",position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                       cv2.putText(frame, f"{distance:.2f} m",(position[0],position[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)

            output_frames.append(frame)
        
        return output_frames # Final list of frames with speed and distance