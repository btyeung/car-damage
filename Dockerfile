FROM lambda-stack:20.04

# Create a non-root user
ARG username="damage-user"
ARG uid=1000
ARG gid=100
ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER
RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --gid $GID \
    --home $HOME \
    $USER

USER $USER    
WORKDIR $HOME

EXPOSE 5000

ENV FLASK_APP=serve_model.py
ENV FLASK_RUN_HOST=0.0.0.0

#transfer source files
COPY './src' $HOME/src/

#RUN apt-get update && \
#    apt-get install -y software-properties-common && \
#    rm -rf /var/lib/apt/lists/*
#RUN add-apt-repository ppa:deadsnakes/ppa
#RUN apt-get update && apt-get install -y python3.8 python3-pip

#install dependencies
RUN pip3 install -r $HOME/src/models/requirements-serve.txt

#TODO: copy in weight file
COPY './models/weights/.pth' $HOME/models/reference.pth

#show python version
RUN python --version

RUN echo "Checking for GPU/CUDA support...1 is good, 0 means no GPU present"
RUN python -c "import torch; print(torch.cuda.device_count())"

#Run app
CMD python $HOME/models/serve_model.py
