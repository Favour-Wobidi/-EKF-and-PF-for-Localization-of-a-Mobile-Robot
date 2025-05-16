# 1. Use a minimal official Python image
FROM python:3.9-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy & install requirements first (this layer will cache until requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your code
COPY . .

# 5. (Optional) Create an outputs directory
RUN mkdir -p results

# 6. Default command: run the particle filter experiments
CMD ["python", "pf_experiments.py"]
