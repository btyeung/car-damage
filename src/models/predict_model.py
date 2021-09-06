# USAGE
# python predict_model.py --model custom-auto --weights models/weights/test-images/66.jpg --image test/12.jpg --labels coco_classes.pickle

# import the necessary packages
from torchvision.models import detection
import numpy as np
import argparse
import pickle
import torch
import cv2


#additional
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


#instantiate the damage model
def get_auto_model(weights_path, num_classes=2):

    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)

    return model

#main section
if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, required=True,
        help="path to the input image")
    ap.add_argument("-w", "--weights", type=str, required=True,
        help="path to the weights file")
    ap.add_argument("-m", "--model", type=str, default="custom-auto",
        choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet", "custom-auto"],
        help="name of the object detection model")
    ap.add_argument("-c", "--confidence", type=float, default=0.10,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # set the device we will be using to run the model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load weights file
    WEIGHTS_PATH = args["weights"]

    #Num classes constant
    NUM_CLASSES = 2

    #TODO: Num of classes is hardcoded
    COLORS = np.random.uniform(0, 255, size=(NUM_CLASSES, 3))

    # initialize a dictionary containing model name and it's corresponding 
    # torchvision function call

    MODELS = {
        "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
        "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
        "retinanet": detection.retinanet_resnet50_fpn,
        "custom-auto": get_auto_model(WEIGHTS_PATH, NUM_CLASSES)
    }

    #TODO: redundant,  load the model and set it to evaluation mode
    if args["model"] != "custom-auto":
        model = MODELS[args["model"]]
    else: 
        model = MODELS[args["model"]]
        #TODO: send to device
        
    #init for inference
    model.eval()

    # load the image from disk
    image = cv2.imread(args["image"])
    orig = image.copy()

    # convert the image from BGR to RGB channel ordering and change the
    # image from channels last to channels first ordering
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))

    # add the batch dimension, scale the raw pixel intensities to the
    # range [0, 1], and convert the image to a floating point tensor
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)

    # send the input to the device and pass the it through the network to
    # get the detections and predictions
    image = image.to(DEVICE)
    detections = model(image)[0]

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
        if confidence > args["confidence"]:
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

    # show the output image
    cv2.imshow("Output", orig)
    cv2.waitKey(0)