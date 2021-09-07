#Makefile for water meter training and inference


#TODO: Run single test, test predictions or inference
run-predict:
	python ./src/models/predict_model.py --model custom-auto --weights ./models/weights/final.pth --image ./references/test-images/66.jpg


#TODO: validate this works (path issues, make sure this resolves from the py file)
start-server:
	python ./src/models/serve_model.py


#TODO: Build docker image for serving model
build-image:
	echo "Building docker container"
	docker build . --tag ai-car-damage

#Run docker image for serving model, CPU only
run-image:
	echo "Running docker container, CPU"
	docker run --publish 5000:5000 ai-car-damage	

#Run docker image for serving model, with GPU (be sure you have this by running nvidia-smi first)
run-image-gpu:
	echo "Running docker container, with GPU support"
	docker run --publish 5000:5000 --gpus all ai-car-damage

#TODO: 
run-tests:
	echo "Running unit tests"
	pytest

