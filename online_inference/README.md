# Homework №2

### _To build image_

From DockerHub:
```
docker pull el3bq02bafg6/made_mlops_hw2:v1
```

From source: from `online_inference/` run:

```
docker build -t el3bq02bafg6/made_mlops_hw2:v1 .
```

### _To run container_
```
docker run --name inference -p 8000:8000 el3bq02bafg6/made_mlops_hw2:v1
```

### _To test service_ 
Open new terminal, from `online_inference/` run:
```
python3 test_inference.py
```
### _To run unit tests_
```
docker exec -it inference bash
python3 -m pytest app_tests.py
```

### _Docker image size reducing_
Replacing the base docker image from python:3.9.12 to python:3.9.12-slim
helped reduce the size of the image by more than 2 times -
https://tinyurl.com/tf7veaz8.

I tried to use python:3.9-alpine, but it didn't work out
