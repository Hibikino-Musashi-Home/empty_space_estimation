FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgl1-mesa-dev libx11-dev && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /gemini_docker/api

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

# CMD ["python3", "gemini_flask_api.py"]
# CMD ["python3", "gemini_flask_preparation_api.py"]

