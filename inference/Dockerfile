# Dockerfile for training
# Start from a base Python image
FROM python:3.8.20

ARG model_name=model.pth
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

WORKDIR /app

# Copy data folder to Docker
COPY data/ /app/data

# Copy models folder to Docker
COPY models/${model_name} /app/models/model.pth

# Copy the code itself
COPY inference /app/inference

# Copy usefull scripts and configs
COPY utils.py /app
COPY ${CONF_PATH} /app

# Install any necessary packages listed in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Train the model
RUN python3 inference/run.py