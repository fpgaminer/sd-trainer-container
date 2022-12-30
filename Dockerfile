FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1 PIP_NO_CACHE_DIR=1

RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

RUN apt-get update && \
	apt-get install -y wget git && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN pip install -U \
	requests \
	pillow \
	transformers[torch] \
	diffusers["torch"] \
	wandb \
	datasets

COPY Train.py /Train.py

ENTRYPOINT ["python", "/Train.py"]