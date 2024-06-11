# Football Analysis Project

## Introduction
In this project we detect and track players, referees and footballs in a video clip using YOLO. I trained a model to improve its perfomance in detecting a ball. I also assigned players to teams based on the colors of their jerseys, using KMeans for pixel segmentation and clustering. Using this, I used logic and math to measure a team's ball acquistion percentage in a match. I also used concepts like interpolation to determine ball position between the frames and fill it in the tracks. Optical flow was used to measure camera movement between the frames, which helped measure a player's movement more accurately. Finally, I implemented perspective transformation to represent a field's depth and perspective, allowing me to measure a player's movement and speed in metres and kilometres per hous, respectively, instead of pixels.

Here's a few screenshots of what the end result looks like.

![Screenshot] [(output_data/final_output_1.png](https://github.com/parthbhangla/football-analysis/blob/main/output_data/final_output_1.png))
![Screenshot] (output_data/final_output_2.png)

## Modules I Used:
I used the following modules in this project:
- YOLO: AI object detection model
- OpenCV: For a lot of output frames and drawing annotations on the frames.
- KMeans: Pixel segmentation and clustering to detect t-shirt color
- Optical Flow: Measure camera movement
- Perspective Transformation: Represent scene depth and perspective
- Math: Speed and distance calculation per player.

## Requirements
To run this project, you need to have the following requirements installed:
- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas
