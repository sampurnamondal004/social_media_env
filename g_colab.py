import sys
import subprocess
from pathlib import Path

# Detect environment
try:
    import google.colab
    IN_COLAB = True
    print("🌐 Running in Google Colab - Perfect!")
except ImportError:
    IN_COLAB = False
    print("💻 Running locally - Nice!")

if IN_COLAB:
    print("\n📦 Cloning OpenEnv repository...")
    subprocess.run(["git", "clone", "https://github.com/meta-pytorch/OpenEnv.git"], check=True)

    print("📚 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "requests"], check=True)

    sys.path.insert(0, "./OpenEnv/src")
    print("\n✅ Setup complete! Everything is ready to go! 🎉")

else:
    sys.path.insert(0, str(Path.cwd().parent / "src"))
    print("✅ Using local OpenEnv installation")

print("\n🚀 Ready to explore OpenEnv and build amazing things!")