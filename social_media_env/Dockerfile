FROM python:3.11-slim

WORKDIR /app

# Copy all files from your current directory to /app in the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e ./OpenEnv

# Set the path so Python can find your modules
ENV PYTHONPATH="/app"

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]