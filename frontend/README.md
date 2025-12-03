# Trash Optimizer Streamlit Frontend

This is the Streamlit frontend for the Trash Optimizer application. It provides an interactive interface for users to visualize and optimize trash collection routes.

## 3rd Party Services
This application relies on the following external services:
- **Geolocation Service**: OpenRouteService for mapping and geolocation functionalities. `https://openrouteservice.org/`
- **Inference Service**: A custom backend service for trash bin level prediction. Make sure to have it running and accessible.

## Setup
```bash
pip install -r requirements.txt
```
## Configuration
1. Copy the `.env.template` file to a new file named `.env`.
2. Create a service account and obtain the necessary API keys for the geolocation and inference services.
3. Fill in the required environment variables in the `.env` file.

## Running the application

```bash
streamlit run app.py
```
