FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04
ENV PYTHONUNBUFFERED 1
ENV TZ "Europe/Moscow"
RUN apt update
RUN apt full-upgrade -y
RUN apt install -y build-essential
RUN mkdir /var/run/sshd

RUN DEBIAN_FRONTEND=noninteractive apt install -y build-essential python3 python3-pip openssh-server wget curl rsync git git-lfs htop mc tmux python-is-python3 tzdata libzbar0

RUN groupadd --gid 1000 user \
  && useradd --uid 1000 --gid user --shell /bin/bash --create-home user

RUN pip install --upgrade setuptools pip
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install jupyter

RUN apt clean
RUN rm -rf /var/lib/apt/lists/*

RUN echo "user:12345678" | chpasswd

WORKDIR /home/user

CMD ["/usr/sbin/sshd", "-D"]
