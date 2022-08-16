docker run --shm-size=8g --name <docker container name> \
-v <folder data-train>:/workspace/data/train/raw/ \
-v <folder public-test>:/workspace/data/public-test/raw/ \
-v <folder for prediction.zip>:/workspace/data/final-output/
<docker image name>

