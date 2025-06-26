# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install uv, the package installer/resolver
RUN pip install uv

# Copy the dependency files and install them
COPY pyproject.toml uv.lock ./
RUN uv sync --no-cache

# Copy the rest of the application code into the container
# This copies server.py, analytics.py, html files, etc.
# .dockerignore will prevent .venv, .git etc from being copied.
COPY . .

# The command to run your application using uv to handle the venv
CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]