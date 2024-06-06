# Importing the library
from ultralytics import YOLO

# Choosing the model
model = YOLO('yolov8x')

# Saving the results
result = model.predict('input_data/08fd33_4.mp4', save = True)

# Printing the results
print(result[0])
print('============================')
for box in result[0].boxes:
    print(box)