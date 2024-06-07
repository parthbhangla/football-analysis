# Importing Libraries
from utilities import read_video, save_video
from tracking import Tracker
from team_assignment import TeamAssigner
from player_ball_assignment import PlayerBallAssigner

def main():
    # Reading Video
    video_frames = read_video('input_data/08fd33_4.mp4')

    # Initializing Tracker
    tracker = Tracker('models/best_yolov8x.pt')

    # Getting the tracks
    tracks = tracker.track(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Interpolate Ball Positions
    tracks['ball'] = tracker.interpolate_ball_position(tracks['ball'])

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0]) # Assigning colors using only the first frame
    
    # Assigning Team to Players for each frame
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team # Team Data
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team] # Team Color Data

    # Assign Ball Acqusition
    player_assigner = PlayerBallAssigner()
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        # If a player is assigned to the ball
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True # New Parameter in the dictionary

    # Drawing annotations on the video
    output_video_frames = tracker.drawing_annotations(video_frames, tracks)

    # Saving The Video
    save_video(output_video_frames, 'output_data/output_video.avi')

if __name__ == '__main__':
    main()