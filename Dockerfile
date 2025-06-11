# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-latex-recommended \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create src directory if it doesn't exist
RUN mkdir -p src

# Expose port for Jupyter notebook
EXPOSE 8888

# Set environment variables
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Start Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
