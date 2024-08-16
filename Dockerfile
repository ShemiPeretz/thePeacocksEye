# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /Engine/api
WORKDIR /server_api/

RUN mkdir "src"

# Copy the entire project into the container at /Engine/api
COPY src ./src
COPY .env ./src

#Copy requirements file
COPY requirements.txt .
#Install requirements
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y iputils-ping
# Make port 80 available to the world outside this container
EXPOSE 80
# Run the FastAPI application in main.py when the container launches
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "80"]