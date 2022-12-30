Test repo

Example usage:

docker run --rm --gpus '"device=1"' -it -v (pwd)/docker-huggingface-cache:/root/.cache/huggingface -e HUGGING_FACE_HUB_TOKEN=some_kind_of_token -e WANDB_API_KEY=some_kind_of_token reddit-nsfw-title-image-pairs-train:latest --device cuda --dataset someuser/somedataset
