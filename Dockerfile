FROM ubuntu:18.04

# Upgrade installed packages
RUN apt update && apt upgrade -y && apt clean

# install python 3.7.10 (or newer)
RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.7 python3.7-dev python3.7-distutils python3-setuptools python3-pip && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Register the version in alternatives (and set higher priority to 3.7)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

RUN update-alternatives --config python3
RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install jupyter
RUN python3 -m pip install jupyter_http_over_ws
RUN jupyter serverextension enable --py jupyter_http_over_ws

RUN python3 -m pip install jupyter_contrib_nbextensions==0.5.1

RUN jupyter nbextension disable contrib_nbextensions_help_item/main && \
    jupyter nbextension disable nbextensions_configurator/config_menu/main && \
    jupyter nbextension enable scroll_down/main && \
    jupyter nbextension enable codefolding/main && \
    jupyter nbextension enable execute_time/ExecuteTime && \
    jupyter nbextension enable toggle_all_line_numbers/main

RUN mkdir -p /home/track_ml
COPY . /home/track_ml
RUN rm -rf /home/track_ml/Dockerfile
WORKDIR /home/track_ml

RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install scikit-image



# docker build --tag python3.7/ubuntu:18.04 .

# docker run --name python37 --publish 127.0.0.1:8080:5678 python3.7/ubuntu:18.04 jupyter notebook --allow-root --ip 0.0.0.0 --port 5678 --NotebookApp.custom_display_url=http://localhost:8080
