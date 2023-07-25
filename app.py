import numpy as np
from flask import Flask, request, app, render_template
import numpy as np
import cv2 as cv
import os
import numpy as np
import os
import torch

UPLOAD_FOLDER = './static/uploads'
app = Flask(__name__, template_folder='./templates', static_folder='./static')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'mlappisrunninghere'


model = torch.hub.load('ultralytics/yolov5', 'custom', path='trained_weights/best.pt')

def draw_bounding_boxes(image, results):
    for detection in results.pandas().xyxy[0].to_dict(orient='records'):
        x1, y1, x2, y2, conf, cls = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax']), float(detection['confidence']), int(detection['class'])
        color = (0, 255, 0)  # Green color for bounding box
        cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image

def detect_drowsiness(image):
    results = model(image)

    # Extracting the detected labels
    labels = results.names  # List of class names
    # print(labels)
    boxes = results.pandas().xyxy[0]  # DataFrame with bounding box coordinates
    # print("Boxes :",boxes)
    label = boxes['name'][0] # label
    # print("label : ",label[0])

    # Draw bounding boxes on the detected image
    detected_img = np.squeeze(results.render())
    detected_img_with_boxes = draw_bounding_boxes(detected_img, results)

    # Save the detected image with bounding boxes in the 'Upload' folder
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_image.jpg')
    cv.imwrite(output_path, detected_img_with_boxes)

    return label


@app.route("/", methods=['GET', 'POST'])
def index():
    data = {"text": "------", "res": False}
    if request.method == "POST":
        try:
            file = request.files['image'].read()
            file_bytes = np.fromstring(file, np.uint8)
            img = cv.imdecode(file_bytes, cv.IMREAD_UNCHANGED)

            # Perform drowsiness detection on the uploaded image
            drowsiness_results = detect_drowsiness(img)
            print("Result :", drowsiness_results)
            # Assuming your drowsiness detection results contain the 'text' and 'res' fields
            data = {"res": drowsiness_results}

            # Display the detection results on the web page
            return render_template("index.html", data=data)
        except:
            print("Exception occured")
            # render_template("error_page.html")
    return render_template("index.html", data=data)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5001, debug=True)
