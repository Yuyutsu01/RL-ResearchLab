FROM python:3.11.8-slim@sha256:90f8795536170fd08236d2ceb74fe7065dbf74f738d8b84bfbf263656654dc9b

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-lock.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-lock.txt

COPY . .

ENTRYPOINT ["python", "main.py"]
CMD ["--config", "configs/cartpole_baseline.yaml", "--mode", "all"]
