# Importing Libraries
from utilities import read_video, save_video
from tracking import Tracker

def main():
    # Reading Video
    video_frames = read_video('input_data/08fd33_4.mp4')

    # Initializing Tracker
    tracker = Tracker('models/best_yolov8x.pt')

    # Getting the tracks
    tracks = tracker.track(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Drawing annotations on the video
    output_video_frames = tracker.drawing_annotations(video_frames, tracks)

    # Saving The Video
    save_video(output_video_frames, 'output_data/output_video.avi')

if __name__ == '__main__':
    main()