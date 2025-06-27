# Use Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Run the app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]
