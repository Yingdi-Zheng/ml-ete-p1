### Notes
When running launcher.sh, one can use parameter:
- `hp` for hyperparameter_tuning functionaility
- `dp` for data_processing functionality
- more to be added


To build locally and test, run:
- `docker build -f Dockerfile -t test-image:1 .` to build the image
- `docker run -d docker.io/library/test-image:1 sleep 300` to get container id that will be able to be tested in 300s 
- `docker exec -it <container id> sh` to execute into the container and be able to run shell commands
- `sh launcher.sh <parameter>` to run function in launcher.sh in the container