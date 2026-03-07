FROM python:3.11-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    liblua5.4-dev \
    libsqlite3-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project
COPY . .

# Build ygopro-core
RUN bash scripts/build_core.sh

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "yugioh_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
