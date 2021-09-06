#Makefile for water meter training and inference


#TODO: Run single test, test predictions or inference
run-predict:
	python ./src/models/predict_model.py --model custom-auto --weights ./models/weights/final.pth --image ./references/test-images/66.jpg --labels ./models/coco_classes.pickle


#TODO: Build docker image for serving model
build-inference-image:
	echo "Building docker container"
	docker build .

#TODO: 
run-tests:
	echo "Running unit tests"
	pytest

#TODO: 
build-docker:
	echo "Building docker container"
	docker build . 

#TODO: resume training from checkpoint
resume-train-checkpoint:
	python ./src/models/train_from_checkpoint.py -f './models/Yolov4_epoch138.pth' -dir './data/raw/train' -epochs 1

#TODO: test this
train-single:
	python ./src/models/train.py -b 2 -s 1 -l 0.001 -g 0 -pretrained ./models/yolov4.conv.137.pth -classes 12 -dir ./data/raw/train -epochs 1