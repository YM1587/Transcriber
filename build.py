import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Ensure PyInstaller is installed before building."""
    try:
        import PyInstaller
        print(f"Found PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller is not installed. Installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

def build_executable():
    """Build the standalone executable."""
    print("Building standalone executable for Free Transcriber...")
    
    # Path to the main application
    app_path = "app.py"
    
    if not os.path.exists(app_path):
        print(f"Error: Could not find main application at {app_path}")
        sys.exit(1)

    # PyInstaller build arguments
    args = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name=FreeTranscriber",
        "--noconfirm",          # Overwrite output directory without confirming
        "--clean",              # Clean PyInstaller cache
        "--onedir",             # Create a one-folder bundle containing an executable
        # Note: --onefile is not recommended for gradio/whisper as it unpacks 
        # hundreds of megabytes on every run, making it extremely slow to start.
        "--collect-all=faster_whisper", 
        "--collect-all=gradio",
        "--collect-all=librosa",
        "--collect-data=gradio_client",
        "--hidden-import=soundfile",
        "--hidden-import=sklearn.utils._typedefs",
        "--hidden-import=sklearn.neighbors._partition_nodes",
        "--hidden-import=sklearn.cluster._agglomerative",
        app_path
    ]

    print(f"Running command: {' '.join(args)}")
    try:
        subprocess.check_call(args)
        print("\n" + "="*50)
        print("Build successful!")
        if sys.platform.startswith("win"):
            print("Executable is located in: dist\\FreeTranscriber\\FreeTranscriber.exe")
        else:
            print("Executable is located in: dist/FreeTranscriber/FreeTranscriber")
        print("="*50)
    except subprocess.CalledProcessError as e:
        print(f"\nBuild failed with error code: {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    check_dependencies()
    build_executable()
