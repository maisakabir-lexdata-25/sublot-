# Frontend Application

This directory contains the Streamlit frontend for the Sublot Intelligence Pro application.

## Prerequisites

Ensure you have the required Python packages installed. From the project root, run:

```bash
pip install -r requirements.txt
```

## Running the Application

To start the frontend, run the following command from the **project root directory** (one level up from this folder):

```bash
streamlit run frontend/app.py
```

### Note on Working Directory
The application expects to be run from the project root directory to correctly locate the model files (e.g., `yolov11_sublot/...`).
