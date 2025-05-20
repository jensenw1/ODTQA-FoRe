## Overview

This project provides a script to run the model service with configurable parameters. Follow the steps below to set up the environment, install dependencies, and execute the script.

---

## Prerequisites

* [Conda](https://docs.conda.io/en/latest/) installed on your system.
* A valid API key for accessing the model backend service.

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

## Usage

To start the prediction, run the following command:

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

