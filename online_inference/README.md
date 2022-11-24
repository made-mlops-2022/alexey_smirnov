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
docker run --name inference -p 8000:8000 lizaavsyannik/online_inference:v5
```

### _To make requests_
Open new terminal, from `online_inference/` run:
```
python3 test_inference.py
```
### _To run tests_
```
docker exec -it inference bash
python3 -m pytest unittests/app_tests.py
```