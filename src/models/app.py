# USAGE
# Start the server:
# 	python serve_meter.py
# Submit a request via cURL:
# 	curl -X POST -F image=@meter.jpg 'http://localhost:5000/predict'

# import the necessary packages
import io

from PIL import Image
import numpy as np
from flask import Flask, send_file, request, Response

import torch
from torch import nn
import torch.nn.functional as F




#TODO: clean up all above imports
from inference import MeterInference

use_cuda = 1

#TODO: convert to config file
weightfile = 'models/reference.pth'

# initialize our Flask application 
app = Flask(__name__)

@app.route("/")
def home():
    return "<h3>AI Water Meter Reading</h3>"

#returns marked up image with bounding boxes drawn
@app.route("/annotate", methods=["POST"])
def annotate():
    #Unclear where I should instantiate the inference object... for performance sake

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            #instantiate the inference runner
            infRunner = MeterInference(use_cuda, weightfile)

            #execute inference
            annotationResults = infRunner.annotateByImage(image)

            # create file-object in memory
            file_object = io.BytesIO()

            #persist file
            annotationResults["image"].save(file_object, 'PNG')

            # move to beginning of file so `send_file()` it will read from start    
            file_object.seek(0)

            print ("Meter read as: " + str(annotationResults["value"]))

            return send_file(file_object, mimetype='image/PNG')

        return None        


@app.route("/read", methods=["POST"])
def read():

    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            #instantiate the inference runner
            infRunner = MeterInference(use_cuda, weightfile)

            #TODO call a predict function for reusability
            results = infRunner.getReadingByImage(image)

            #get the reading from the boxes DF			
            data["reading"] = results

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return data


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading meter reading server..." "please wait until server has fully started"))

    #TODO: change this to be more secure
    app.run(host='0.0.0.0', port=5000)