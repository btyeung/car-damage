# USAGE
# Start the server:
# 	python serve_model.py
# Submit a request via cURL:
# 	curl -X POST -F image=@damage.jpg 'http://localhost:5000/annotate'

# import the necessary packages
import io
import os

from PIL import Image
import numpy as np
from flask import Flask, send_file, request, Response, make_response

import torch
from torch import nn
import torch.nn.functional as F
import cv2

import predict_model as predict

#get current working directory
cwd = os.getcwd()

print ("Current working directory: " + cwd)

#default to CPU, unless GPU detected on start
DEVICE = torch.device("cpu")

#classes hardcode
NUM_CLASSES = 2
CONFIDENCE = 0.20

WEIGHTFILE_PATH = './models/weights/final.pth'

# initialize our Flask application 
app = Flask(__name__)

@app.route("/")
def home():
    return "<h3>Auto Damage Detector</h3>"

#returns marked up image with bounding boxes drawn
@app.route("/annotate", methods=["POST"])
def annotate():

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("image"):

            # read the image in PIL format
            raw_image = request.files["image"].read()            

            nparr = np.fromstring(raw_image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            orig = img.copy()

            #convert image to tensor and pass back
            image_tensor = predict.image_to_tensor(img)

            #num classes hardcoded for random color usage
            COLORS = np.random.uniform(0, 255, size=(NUM_CLASSES, 3))

            image_tensor = image_tensor.to(DEVICE)

            # perform inference
            detections = model(image_tensor)[0]

            #sanity check
            print ("Detections: " + str(len(detections["boxes"])))

            # loop over the detections
            for i in range(0, len(detections["boxes"])):
                    
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections["scores"][i]
                print ("Score: " + str(confidence))

                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > CONFIDENCE:
                    # extract the index of the class label from the detections,
                    # then compute the (x, y)-coordinates of the bounding box
                    # for the object
                    idx = int(detections["labels"][i])
                    box = detections["boxes"][i].detach().cpu().numpy()
                    (startX, startY, endX, endY) = box.astype("int")

                    # display the prediction to our terminal
                    #TODO: remove hardcode of classname
                    #label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    label = "{}: {:.2f}%".format("Damage", confidence * 100)
                    print("[INFO] {}".format(label))

                    # draw the bounding box and label on the image
                    cv2.rectangle(orig, (startX, startY), (endX, endY),
                        COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(orig, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            retval, buffer = cv2.imencode('.png', orig)
            response = make_response(buffer.tobytes())
            response.headers['Content-Type'] = 'image/png'

            return response

    return None
    
# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading AI damage  server..." "please wait until server has fully started"))

    #determine if GPU used
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #instantiate the model
    model = predict.get_auto_model(WEIGHTFILE_PATH, DEVICE, 2)

    #init for inference
    model.to(DEVICE)
    model.eval()

    #TODO: change this to be more secure
    app.run(host='0.0.0.0', port=5000)