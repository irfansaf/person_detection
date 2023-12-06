from ultralytics import YOLO

model = YOLO('yolov8m.yaml')  # build a new model from YAML
results = model.train(data='datasets/data.yaml', epochs=300, batch=16, imgsz=640, verbose=True)
#model = YOLO('runs/detect/train2/weights/best.pt')

results = model.val()  # run test.py automatically with model 'best.pt'

results = model ("test.jpg")

success = model.export(format='onnx')