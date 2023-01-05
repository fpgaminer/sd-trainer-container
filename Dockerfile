#FROM python:3.10-slim
#FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
# TODO: I had to switch to pytorch/pytorch to support bitsandbytes, since bitsandbytes needed cudart.
# TODO: xformers doesn't work. Even when I get it installed correctly, it errors out with unimplemented attention ops.
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1 PIP_NO_CACHE_DIR=1

#RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

RUN apt-get update && \
	apt-get install -y wget git && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

# TODO: https://github.com/TimDettmers/bitsandbytes/issues/85
#RUN pip install --extra-index-url https://pypi.ngc.nvidia.com nvidia-cuda-runtime-cu11

RUN pip install -U \
	requests==2.28.1 \
	Pillow==9.3.0 \
	transformers==4.25.1 \
	diffusers==0.11.1 \
	wandb==0.13.7 \
	datasets==2.8.0 \
	bitsandbytes==0.35.4 \
	accelerate==0.15.0

COPY Train.py /

ENTRYPOINT ["python", "/Train.py"]