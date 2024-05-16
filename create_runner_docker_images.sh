docker build . -f Dockerfile.runner -t gitlab.assist-iot.eu:5050/wp7/pilot-1/container-detection:1.0.0-gpu --build-arg VARIANT=-gpu
docker build . -f Dockerfile.runner -t gitlab.assist-iot.eu:5050/wp7/pilot-1/container-detection:1.0.0-cpu --build-arg VARIANT=

docker build . -f Dockerfile.runner_with_video -t gitlab.assist-iot.eu:5050/wp7/pilot-1/container-detection:1.0.0-cpu-video --build-arg TAG=1.0.0-cpu
docker build . -f Dockerfile.runner_with_video -t gitlab.assist-iot.eu:5050/wp7/pilot-1/container-detection:1.0.0-gpu-video --build-arg TAG=1.0.0-gpu

docker push gitlab.assist-iot.eu:5050/wp7/pilot-1/container-detection:1.0.0-gpu
docker push gitlab.assist-iot.eu:5050/wp7/pilot-1/container-detection:1.0.0-cpu
docker push gitlab.assist-iot.eu:5050/wp7/pilot-1/container-detection:1.0.0-gpu-video
docker push gitlab.assist-iot.eu:5050/wp7/pilot-1/container-detection:1.0.0-cpu-video
