# Importing Libraries
from utilities import read_video, save_video

def main():
    # Reading Video
    video_frames = read_video('YOUR_VIDEO_PATH')

    # Saving The Video
    save_video(video_frames, 'output_data/output_video.avi')

if __name__ == '__main__':
    main()