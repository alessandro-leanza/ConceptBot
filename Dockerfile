FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1     PYTHONPATH=/workspace

WORKDIR /workspace

RUN apt-get update     && apt-get install -y --no-install-recommends build-essential     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip     && python -m pip install -r /tmp/requirements.txt

CMD ["python", "instructions/load_instructions.py"]
