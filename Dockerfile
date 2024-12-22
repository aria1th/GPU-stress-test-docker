# Use a base image with CUDA support
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip

# Install CuPy and other dependencies
RUN pip3 install cupy-cuda11x  # Install CuPy for CUDA 11.x (adjust if you use a different CUDA version)

# Install other useful tools for debugging
RUN apt-get install -y build-essential gdb

# Create a working directory
WORKDIR /app

# Copy the Python stress test script
COPY stress_test.py /app/

# Expose a command for running the stress test (optional, for convenience)
CMD ["python3", "stress_test.py"]