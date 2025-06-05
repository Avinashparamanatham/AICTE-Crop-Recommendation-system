import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Install all required packages"""
    print("🌾 Installing Crop Recommendation System Requirements")
    print("=" * 55)
    
    requirements = [
        "streamlit==1.28.1",
        "pandas==2.0.3", 
        "numpy==1.24.3",
        "scikit-learn==1.3.0",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "plotly==5.17.0",
        "Pillow==10.0.1"
    ]
    
    failed_packages = []
    
    for package in requirements:
        print(f"📦 Installing {package}...")
        if install_package(package):
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ Failed to install {package}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n❌ Failed to install: {', '.join(failed_packages)}")
        print("Please install them manually using:")
        for package in failed_packages:
            print(f"   pip install {package}")
        return False
    else:
        print("\n✅ All packages installed successfully!")
        return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)