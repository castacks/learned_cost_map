# FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04 # Didn't work. Gave me early error and exit code 100 for first RUN command
FROM nvidia/cuda:11.3.0-devel-ubuntu18.04 
#11.4.0-cudnn8-devel-ubuntu18.04 # This one works
# FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04 # Didn't work with tzdata

# useful tools
RUN apt-get update \
 && apt-get install -y \
        build-essential \
        cmake \
        cppcheck \
        gdb \
        git \
        sudo \
        vim \
        wget \
        tmux \
        curl \
        less \
        htop \
        libsm6 libxext6 libgl1-mesa-glx libxrender-dev \
 && apt-get clean

# Add a user with the same user_id as the user outside the container
# Requires a docker build argument `user_id`
ARG user_id=1132
ARG group_id=403
RUN groupadd -g ${group_id} frc_members
ENV USERNAME mguamanc
RUN useradd --uid ${user_id} --gid ${group_id} -ms /bin/bash $USERNAME \
 && echo "$USERNAME:$USERNAME" | chpasswd \
 && adduser $USERNAME sudo \
 && echo "$USERNAME ALL=NOPASSWD: ALL" >> /etc/sudoers.d/$USERNAME

# run as the developer user
USER $USERNAME

# running container start dir
WORKDIR /home/$USERNAME

# allow using GUI apps
RUN export DEBIAN_FRONTEND=noninteractive \
 && sudo apt-get update \
 && sudo -E apt-get install -y \
    tzdata \
 && sudo ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime \
 && sudo dpkg-reconfigure --frontend noninteractive tzdata \
 && sudo apt-get clean

# useful tools
RUN sudo apt-get update \
 && sudo apt-get install -y python-tk python3-pip \
 && sudo apt-get clean

# Python 3.
RUN sudo pip3 install --upgrade pip
RUN sudo pip3 install --no-cache-dir numpy scipy matplotlib ipython pandas visdom scikit-image scikit-learn opencv-python opencv-contrib-python numba ninja Pillow colorcet plyfile sklearn tqdm
# RUN sudo pip3 install --no-cache-dir cupy-cuda110
RUN sudo pip3 install --no-cache-dir cupy-cuda113

# Pillow version is fixed for torchvision.

RUN sudo pip3 install --no-cache-dir torch==1.10.0 torchvision # tensorflow
RUN sudo pip3 install --no-cache-dir pyyaml gym tabulate

RUN sudo pip3 install --no-cache-dir catkin_pkg


# entrypoint command
CMD /bin/bash
