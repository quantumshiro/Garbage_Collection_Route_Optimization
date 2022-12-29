FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project
COPY src/ /app/src/
COPY data/garbage_place.xlsx /app/data/garbage_place.xlsx


# Run server use uvicorn
CMD [ "uvicorn", "app.src.app:app", "--reload"]
