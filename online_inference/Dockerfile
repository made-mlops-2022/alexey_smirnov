FROM python:3.9.12-slim

WORKDIR /online_inference
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir
COPY app.py schemas.py run.sh config.py unittests/app_tests.py ./
ENV TRANSFORMER_PATH="transformer.pkl"
ENV MODEL_PATH="output_model.pkl"
EXPOSE 8000
CMD ["bash", "run.sh"]
