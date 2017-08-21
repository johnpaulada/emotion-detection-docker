FROM ubuntu:latest
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y build-essential
RUN apt-get install -y cmake
RUN apt-get install -y python-dev
RUN apt-get install -y python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
RUN apt install -y python3-pip
RUN apt-get install -y libboost-all-dev
RUN pip3 install --upgrade pip
RUN pip3 install dlib
RUN pip3 install jupyter
RUN pip3 install scikit-image
RUN pip3 install opencv-python
RUN pip3 install --upgrade imutils
RUN pip3 install -U scikit-learn
WORKDIR /home
ENTRYPOINT /bin/bash