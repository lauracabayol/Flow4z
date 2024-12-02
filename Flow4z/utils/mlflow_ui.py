import subprocess

def start_mlflow_ui():
    """Start the MLflow UI."""
    subprocess.Popen(["mlflow", "ui"])