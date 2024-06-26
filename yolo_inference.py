# Importing the library
from ultralytics import YOLO

# Choosing the model
model = YOLO('MODEL_PATH_HERE')

# Saving the results
result = model.predict('INPUT_DATA_PATH_HERE', save = True) # Put your model path here

# Printing the results
print(result[0])
print('============================')
for box in result[0].boxes:
    print(box)