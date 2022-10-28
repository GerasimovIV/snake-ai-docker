from jerinka/opencv:1
from nvcr.io/nvidia/pytorch:22.01-py3
ARG DEBIAN_FRONTEND=noninteractive
RUN pip install termcolor
RUN pip install pygame
RUN apt-get update && apt-get -y install libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev libportmidi-dev libfreetype6-dev libavformat-dev libswscale-dev mercurial
COPY . /workspace
ENTRYPOINT ["/bin/bash"]
