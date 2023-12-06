from numpy import asarray
from ultralytics import YOLO
from flask import Flask, request, Response
from PIL import Image
from mtcnn.mtcnn import MTCNN
import json

app = Flask(__name__)

@app.route('/')
def root():
    with open('webservices/index.html') as f:
        return f.read()

@app.route('/detect', methods=['POST'])
def detect():
    buf = request.files['image_file']

    image_array = asarray(Image.open(buf))
    pil_image = Image.fromarray(image_array)

    face_boxes = detect_faces_on_image(image_array)
    recognized_faces = recognize_faces(pil_image, face_boxes)
    all_boxes = face_boxes + recognized_faces

    print(all_boxes)
    return Response(
        json.dumps(all_boxes),
        mimetype='application/json'
    )

def detect_faces_on_image(image_array):
    detector = MTCNN()
    results = detector.detect_faces(image_array)

    output = []
    for result in results:
        x1, y1, width, height = result['box']
        x2 = x1 + width
        y2 = y1 + height
        prob = round(result['confidence'], 2)
        output.append([
            x1, y1, x2, y2, 'face', prob
        ])

    return output

def recognize_faces(image, face_boxes):
    recognized_faces = []
    model = YOLO('v5model.pt')

    for face_box in face_boxes:
        x1, y1, x2, y2, _, _ = face_box
        face_region = image.crop((x1, y1, x2, y2))

        yolo_format_box = [
            x1 / image.width, y1 / image.height, x2 / image.width, y2 / image.height
        ]

        results = model.predict(source=face_region)
        yolo_result = results[0]

        if len(yolo_result.boxes) > 0:
            detected_class_id = yolo_result.boxes[0].cls[0].item()
            detected_class_name = yolo_result.names[detected_class_id]
            prob = round(yolo_result.boxes[0].conf[0].item(), 2)

            recognized_faces.append({
                'box': face_box,
                'name': detected_class_name,
                'confidence': prob
            })
        else:
            recognized_faces.append({
                'box': face_box,
                'name': 'Unknown',
                'confidence': 0.0
            })

    return recognized_faces

def resize_image(image, max_width, max_height):
    width, height = image.size
    aspect_ratio = width / height

    if width > max_width or height > max_height:
        if aspect_ratio > 1:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)

        image = image.resize((new_width, new_height))

    return image

if __name__ == '__main__':
    app.run(debug=True)
