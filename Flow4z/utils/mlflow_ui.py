import subprocess

def start_mlflow_ui():
    """Start the MLflow UI if not already running."""
    # Check if mlflow UI is already running by trying to connect to default port
    try:
        import requests
        requests.get("http://127.0.0.1:5000")
        return None  # UI already running
    except requests.exceptions.ConnectionError:
        # Start mlflow UI since it's not running
        process = subprocess.Popen(["mlflow", "ui"], cwd="../..")
        return process

def stop_mlflow_ui(process):
    """Stop the MLflow UI.
    
    Args:
        process: The process object returned by start_mlflow_ui()
    """
    if process:
        process.terminate()
        process.wait()