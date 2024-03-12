# Use python:3.9-slim as the base image
FROM python:3.9-slim

# Set an environment variable to ensure Python output is sent straight to the terminal
# without being first buffered and that you see the output of your application in real time
ENV PYTHONUNBUFFERED True

# Set the working directory in the container to /app
ENV APP_HOME /app
WORKDIR $APP_HOME

# Install system dependencies required by LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements.txt file into our directory (to cache the installed packages)
COPY requirements.txt ./

# Upgrade pip and install dependencies from requirements.txt
# This layer will be rebuilt only if requirements.txt changes
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the code into the container
COPY . ./

CMD cd src && python training.py