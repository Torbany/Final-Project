# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Create a new user
RUN useradd -ms /bin/bash appuser

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Upgrade pip
RUN pip install --upgrade pip

# Copy the rest of the application code
COPY . .

# Change ownership of the application folder
RUN chown -R appuser:appuser /app

# Switch to the new user
USER appuser

# Expose the port the app runs on
EXPOSE 5000

# Set environment variables to ensure Python prints directly to the terminal
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "app.py"]
