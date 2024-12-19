# Step 1: Use an official Python runtime as the base image
FROM python:3.10-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current project structure into the container
COPY . .

# Step 4: Upgrade pip
RUN python -m pip install --upgrade pip

# Step 5: Install project dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Define the command to run classifiers.py followed by main.py
CMD ["bash", "-c", "python classifiers.py && python main.py"]
