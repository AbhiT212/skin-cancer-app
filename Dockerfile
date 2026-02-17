# Use a lightweight Python image to keep the container small
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

RUN cat weights/best_model.pth.part* > weights/best_model.pth && rm weights/best_model.pth.part*
# Expose the port that Google Cloud Run expects
EXPOSE 8080

# Command to run your Dash web app using Gunicorn
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120"]