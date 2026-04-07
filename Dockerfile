# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements/project files first (to cache layers)
COPY OpenEnv/ /app/OpenEnv/
COPY OpenEnv/model.py OpenEnv/reward.py OpenEnv/sm_env.py /app/
RUN pip install --no-cache-dir -e ./OpenEnv
ENV PYTHONPATH="/app"
# Install the dependencies and the local package
RUN pip install --no-cache-dir -e ./OpenEnv

# Command to run your script
CMD ["python", "model.py"]