FROM python:3.9.12-slim

WORKDIR /online_inference
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir
COPY . .
ENV TRANSFORMER_PATH="transformer.pkl"
ENV MODEL_PATH="model.pkl"
EXPOSE 8000
CMD ["bash", "run.sh"]