#!/bin/bash

# Docker setup script for Python notebook project
# This script creates and runs a Docker container with Jupyter notebook

set -e  # Exit on any error

PROJECT_NAME="python-notebook-project"
CONTAINER_NAME="notebook-container"
IMAGE_NAME="notebook-image"
PORT=8888

echo "=== Docker Setup for Python Notebook Project ==="

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "Error: Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to create Dockerfile
create_dockerfile() {
    echo "Creating Dockerfile..."
    cat > Dockerfile << 'EOF'
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
EOF
}

# Function to build Docker image
build_image() {
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME .
}

# Function to run container
run_container() {
    echo "Starting container..."
    
    # Stop existing container if running
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        echo "Stopping existing container..."
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
    fi
    
    # Get local IP address
    LOCAL_IP=$(hostname -I | awk '{print $1}')
    
    # Run new container - bind to all interfaces for network access
    docker run -d \
        --name $CONTAINER_NAME \
        -p 0.0.0.0:$PORT:8888 \
        -v "$(pwd)":/app \
        $IMAGE_NAME
    
    echo "Container started successfully!"
    echo "Jupyter notebook is available at:"
    echo "  Local access:   http://localhost:$PORT"
    echo "  Network access: http://$LOCAL_IP:$PORT"
    echo ""
    echo "Other devices on your network can access it using: http://$LOCAL_IP:$PORT"
    echo ""
    echo "To view container logs: docker logs $CONTAINER_NAME"
    echo "To stop container: docker stop $CONTAINER_NAME"
    echo "To access container shell: docker exec -it $CONTAINER_NAME bash"
}

# Function to convert notebook to PDF
convert_to_pdf() {
    if [ -z "$2" ]; then
        echo "Usage: $0 pdf <notebook.ipynb> [output.pdf]"
        echo "Example: $0 pdf my_notebook.ipynb my_report.pdf"
        exit 1
    fi
    
    NOTEBOOK_FILE="$2"
    OUTPUT_FILE="${3:-${NOTEBOOK_FILE%.ipynb}.pdf}"
    
    if [ ! -f "$NOTEBOOK_FILE" ]; then
        echo "Error: Notebook file '$NOTEBOOK_FILE' not found"
        exit 1
    fi
    
    echo "Converting $NOTEBOOK_FILE to PDF..."
    docker exec -it $CONTAINER_NAME jupyter nbconvert --to pdf "/app/$NOTEBOOK_FILE" --output "/app/$OUTPUT_FILE"
    
    if [ $? -eq 0 ]; then
        echo "Successfully converted to: $OUTPUT_FILE"
    else
        echo "Error: PDF conversion failed"
        echo "Trying alternative method with HTML intermediate..."
        docker exec -it $CONTAINER_NAME jupyter nbconvert --to html "/app/$NOTEBOOK_FILE"
        echo "HTML file created. You can print this to PDF from your browser."
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [build|run|stop|restart|logs|shell|pdf]"
    echo ""
    echo "Commands:"
    echo "  build    - Build the Docker image"
    echo "  run      - Run the container"
    echo "  stop     - Stop the container"
    echo "  restart  - Restart the container"
    echo "  logs     - Show container logs"
    echo "  shell    - Access container shell"
    echo "  pdf      - Convert notebook to PDF"
    echo ""
    echo "PDF Usage: $0 pdf <notebook.ipynb> [output.pdf]"
    echo ""
    echo "If no command is provided, it will build and run automatically."
}

# Main script logic
case "${1:-auto}" in
    "build")
        check_docker
        create_dockerfile
        build_image
        ;;
    "run")
        check_docker
        run_container
        ;;
    "stop")
        echo "Stopping container..."
        docker stop $CONTAINER_NAME 2>/dev/null || echo "Container not running"
        docker rm $CONTAINER_NAME 2>/dev/null || echo "Container not found"
        ;;
    "restart")
        check_docker
        $0 stop
        $0 run
        ;;
    "logs")
        docker logs $CONTAINER_NAME
        ;;
    "shell")
        docker exec -it $CONTAINER_NAME bash
        ;;
    "pdf")
        convert_to_pdf "$@"
        ;;
    "auto")
        check_docker
        create_dockerfile
        build_image
        run_container
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        echo "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac