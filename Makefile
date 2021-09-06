#Makefile for water meter training and inference


#TODO: Run single test, test predictions or inference
run-predict:
	python ./src/models/predict_model.py --model custom-auto --weights ./models/weights/final.pth --image ./references/test-images/66.jpg


#TODO: validate this works (path issues, make sure this resolves from the py file)
start-server:
	python ./src/models/serve_model.py


#TODO: Build docker image for serving model
build-inference-image:
	echo "Building docker container"
	docker build .

#TODO: 
run-tests:
	echo "Running unit tests"
	pytest

