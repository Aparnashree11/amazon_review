# Use an official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p models/distilbert-final
# Copy project files
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Set the command to start script
COPY start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]
