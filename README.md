## Overview

This project provides a script to run the model service with configurable parameters. Follow the steps below to set up the environment, install dependencies, start the necessary database container, and execute the script.

---

## Prerequisites

* [Conda](https://docs.conda.io/en/latest/) installed on your system.
* A valid API key for accessing the model backend service.
* [Docker](https://docs.docker.com/) installed and running.

---

## Database Setup

Before running the prediction script, you need to start a PostgreSQL container using Docker and load all the necessary tables into it.

1. **Start the PostgreSQL container**:

   ```bash
   docker run -id \
     --name=tqa-postgres \
     -v ./data:/var/lib/postgresql/data \
     -p 5432:5432 \
     -e POSTGRES_PASSWORD='123456' \
     -e POSTGRES_USER='tqa' \
     -e LANG=C.UTF-8 \
     --restart=always \
     postgres:alpine
   ```
After successful execution, the database will be accessible via port 5432.

2. **Import all tables into the database**:

   The automation script for importing tables is provided as a Python notebook. Run it to populate the database:

   ```bash
   datasets/tables/import_table.ipynb
   ```

---

## Installation

1. Create and activate a Conda environment:

   ```bash
   conda create -n tqa python=3.10.5 -y
   conda activate tqa
   ```

2. Install required Python packages from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

---

## Reproduction

To reproduce the results, run the following command:

```bash
bash scripts/run.sh MODEL_NAME WORKERS BASE_URL API_KEY
```

**Parameter order is important and must be provided exactly as shown above.**

### Parameters

1. **MODEL\_NAME**: The name or identifier of the model you want to serve.

   * Example: `qwen2.5-7b`, `qwen2.5-72b`

2. **WORKERS**: The number of worker processes to spawn for handling requests.

   * Example: `4`, `8`

3. **BASE\_URL**: The base URL of the model backend service.

   * Example: `http://localhost:8000`, `https://api.yourmodel.com`

4. **API\_KEY**: Your API key for authenticating with the model backend.

   * Example: `sk-XXXXXXXXXXXXXXXXXXXX`

---

## Example

```bash
bash scripts/run.sh qwen2.5-72b 4 http://localhost:8000 sk-1234567890abcdef
```

This will start the prediction using the `qwen2.5-72b` model with 4 worker processes, connecting to `http://localhost:8000` using the provided API key.
