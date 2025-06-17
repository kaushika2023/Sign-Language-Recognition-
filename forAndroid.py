from ultralytics import YOLO

model = YOLO("D:\\pycharm_projects\\epicYolo\\runs\\detect\\train6\\weights\\best.pt")

model.export(format="tflite")