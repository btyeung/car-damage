#Makefile for water meter training and inference


#Run single test
test-one:
	python ./src/detect_image.py --model custom-auto --weights ./models/weights/48.pth --image ./references/test/66.jpg --labels ./models/coco_classes.pickle


#Start here
test-inference:
	python src/models/inference.py

#Build docker image for serving model
build-inference-image:
	echo "Building docker container"
	docker build .

run-tests:
	echo "Running unit tests"
	pytest

build-docker:
	echo "Building docker container"
	docker build . 

#resume training from checkpoint
resume-train-checkpoint:
	python ./src/models/train_from_checkpoint.py -f './models/Yolov4_epoch138.pth' -dir './data/raw/train' -epochs 1

#TODO: test this
train-single:
	python ./src/models/train.py -b 2 -s 1 -l 0.001 -g 0 -pretrained ./models/yolov4.conv.137.pth -classes 12 -dir ./data/raw/train -epochs 1