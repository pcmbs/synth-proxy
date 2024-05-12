#!/bin/bash 

# Docker image name to create container from
IMAGE_NAME="synth-proxy"

# Docker container name
CONTAINER_NAME="$IMAGE_NAME"

# Check custom container name are provided
if [ $# -gt 0 ]; then
    CONTAINER_NAME="$1"
fi

# Check if the container exists and start or create it accordingly
if [ "$(docker ps -q -f name="$CONTAINER_NAME")" ] || [ "$(docker ps -aq -f status=exited -f name="$CONTAINER_NAME")" ] ; then
    echo "Container $CONTAINER_NAME exists, starting and opening shell..."
    docker start -ia "$CONTAINER_NAME" 

else
    echo "Container $CONTAINER_NAME from image $IMAGE_NAME does not exist, creating and starting..."
	docker run -it -d --network=host --gpus all --shm-size=8g --name "$CONTAINER_NAME" \
        --env-file $(pwd)/.env \
        --mount type=bind,src=$(pwd)/checkpoints,dst=/workspace/checkpoints \
        --mount type=bind,src=$(pwd)/configs,dst=/workspace/configs \
        --mount type=bind,src=$(pwd)/data,dst=/workspace/data \
        --mount type=bind,src=$(pwd)/logs,dst=/workspace/logs \
        --mount type=bind,src=$(pwd)/results,dst=/workspace/results \
        --mount type=bind,src=$(pwd)/src,dst=/workspace/src \
        --mount type=bind,src=$(pwd)/tests,dst=/workspace/tests \
        "$IMAGE_NAME:local" 
	
    echo "Waiting for container to start..."
    sleep 5

	echo "Installing Python local packages..."
	docker exec "$CONTAINER_NAME" pip install -e .
	
    echo "Opening shell in container..."
    docker exec -it "$CONTAINER_NAME" bash
fi

echo "Stopping Container..."
docker stop "$CONTAINER_NAME"