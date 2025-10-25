#!/usr/bin/env python3
"""
SecureRAG Installer
One-click installation script for SecureRAG v1.0
"""

import sys
import os
import platform
import subprocess
import json
import shutil
from pathlib import Path


class SecureRAGInstaller:
    """One-click installer for SecureRAG"""

    def __init__(self):
        self.install_dir = Path.home() / ".secure-rag"
        self.project_dir = Path(__file__).parent.absolute()
        self.system = platform.system()

    def print_header(self):
        """Print welcome header"""
        print("\n" + "=" * 60)
        print("  SecureRAG v1.0 Installer")
        print("  Local, Privacy-First RAG for Claude Desktop")
        print("=" * 60 + "\n")

    def check_python_version(self):
        """Check Python version >= 3.10"""
        print("📋 Checking Python version...")

        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 10):
            print(f"   ✗ Python 3.10+ required (found {version.major}.{version.minor})")
            print("\n   Please install Python 3.10 or higher:")
            print("   https://www.python.org/downloads/")
            return False

        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True

    def create_directories(self):
        """Create necessary directories"""
        print("\n📁 Creating directories...")

        directories = [
            self.install_dir,
            self.install_dir / "vector_db",
            self.install_dir / "backups",
            self.install_dir / "models",
            self.install_dir / "logs",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"   ✓ {directory}")

        return True

    def create_venv(self):
        """Create virtual environment"""
        print("\n🐍 Creating virtual environment...")

        venv_dir = self.install_dir / "venv"

        if venv_dir.exists():
            print("   ℹ Virtual environment already exists")
            return True

        try:
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                check=True,
                capture_output=True
            )
            print(f"   ✓ Created at {venv_dir}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ✗ Failed to create virtual environment")
            print(f"   Error: {e.stderr.decode()}")
            return False

    def get_venv_python(self):
        """Get path to virtual environment Python"""
        if self.system == "Windows":
            return str(self.install_dir / "venv" / "Scripts" / "python.exe")
        else:
            return str(self.install_dir / "venv" / "bin" / "python")

    def install_dependencies(self):
        """Install Python dependencies"""
        print("\n📦 Installing dependencies...")
        print("   (This may take several minutes)")

        venv_python = self.get_venv_python()
        requirements_file = self.project_dir / "requirements.txt"

        if not requirements_file.exists():
            print("   ✗ requirements.txt not found")
            return False

        try:
            # Upgrade pip first
            subprocess.run(
                [venv_python, "-m", "pip", "install", "--upgrade", "pip"],
                check=True,
                capture_output=True
            )

            # Install requirements
            subprocess.run(
                [venv_python, "-m", "pip", "install", "-r", str(requirements_file)],
                check=True,
                capture_output=True
            )

            print("   ✓ Dependencies installed")
            return True

        except subprocess.CalledProcessError as e:
            print(f"   ✗ Failed to install dependencies")
            print(f"   Error: {e.stderr.decode()}")
            return False

    def download_models(self):
        """Download ML models"""
        print("\n🤖 Downloading models...")
        print("   (This will download ~2.5 GB)")

        response = input("\n   Download models now? (Recommended) [Y/n]: ")
        if response.lower() == 'n':
            print("   ⊘ Skipped. You can download later with:")
            print(f"     {self.get_venv_python()} scripts/download_models.py")
            return True

        venv_python = self.get_venv_python()
        download_script = self.project_dir / "scripts" / "download_models.py"

        if not download_script.exists():
            print("   ✗ download_models.py not found")
            return False

        try:
            subprocess.run(
                [venv_python, str(download_script), "--models-dir", str(self.install_dir / "models")],
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ✗ Failed to download models: {e}")
            return False

    def init_databases(self):
        """Initialize SQLite databases"""
        print("\n💾 Initializing databases...")

        # Databases will be created automatically on first use
        print("   ✓ Database initialization deferred to first use")
        return True

    def create_config(self):
        """Create configuration file"""
        print("\n⚙️  Creating configuration...")

        config_path = self.install_dir / "config.yaml"

        if config_path.exists():
            print("   ℹ Configuration already exists")
            return True

        # Copy default config
        default_config = self.project_dir / "config" / "default_config.yaml"

        if default_config.exists():
            shutil.copy(default_config, config_path)
            print(f"   ✓ Created at {config_path}")
        else:
            print("   ⚠ Default config not found, will use built-in defaults")

        return True

    def configure_claude_desktop(self):
        """Update Claude Desktop configuration"""
        print("\n🔧 Configuring Claude Desktop...")

        # Determine config path based on OS
        if self.system == "Darwin":  # macOS
            config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        elif self.system == "Windows":
            config_path = Path(os.getenv("APPDATA")) / "Claude" / "claude_desktop_config.json"
        else:  # Linux
            config_path = Path.home() / ".config" / "Claude" / "claude_desktop_config.json"

        # Create directory if needed
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing config or create new
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {}

        # Ensure mcpServers section exists
        if "mcpServers" not in config:
            config["mcpServers"] = {}

        # Get python path
        venv_python = self.get_venv_python()

        # Add SecureRAG server
        config["mcpServers"]["secure-rag"] = {
            "command": venv_python,
            "args": [str(self.project_dir / "src" / "mcp_server.py")],
            "env": {
                "CONFIG_PATH": str(self.install_dir / "config.yaml")
            }
        }

        # Save config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"   ✓ Updated {config_path}")
        print("\n   ⚠️  IMPORTANT: Restart Claude Desktop for changes to take effect")

        return True

    def test_installation(self):
        """Test the installation"""
        print("\n🧪 Testing installation...")

        venv_python = self.get_venv_python()

        # Test imports
        test_script = f"""
import sys
try:
    from src.config import load_config
    from src.vector_store import VectorStore
    from src.embeddings import EmbeddingHandler
    print("✓ All imports successful")
    sys.exit(0)
except Exception as e:
    print(f"✗ Import error: {{e}}")
    sys.exit(1)
"""

        test_file = self.install_dir / "test_install.py"
        test_file.write_text(test_script)

        try:
            result = subprocess.run(
                [venv_python, str(test_file)],
                cwd=str(self.project_dir),
                capture_output=True,
                text=True,
                timeout=30
            )

            print(f"   {result.stdout.strip()}")

            test_file.unlink()

            return result.returncode == 0

        except Exception as e:
            print(f"   ✗ Test failed: {e}")
            return False

    def print_summary(self):
        """Print installation summary"""
        print("\n" + "=" * 60)
        print("✅ Installation Complete!")
        print("=" * 60)
        print(f"\n📁 Installation directory: {self.install_dir}")
        print(f"📝 Configuration file: {self.install_dir / 'config.yaml'}")
        print(f"💾 Vector database: {self.install_dir / 'vector_db'}")
        print(f"📦 Backups: {self.install_dir / 'backups'}")

        print("\n⚠️  NEXT STEPS:")
        print("\n1. Restart Claude Desktop")
        print("2. SecureRAG tools will be available in Claude")
        print("3. Try: 'Create a collection called MyDocs'")
        print("4. Then: 'Ingest /path/to/document.pdf into MyDocs'")
        print("5. Then: 'Search MyDocs for information about...'")

        print("\n📚 Documentation:")
        print("   README: " + str(self.project_dir / "README.md"))

        print("\n💡 Tips:")
        print("   • All data stays local (100% private)")
        print("   • Use KB-only mode to restrict Claude to your documents")
        print("   • Export collections for backup")
        print("   • Version your documents for change tracking")

        print("\n" + "=" * 60 + "\n")

    def run(self):
        """Run the complete installation"""
        self.print_header()

        steps = [
            ("Checking Python version", self.check_python_version),
            ("Creating directories", self.create_directories),
            ("Creating virtual environment", self.create_venv),
            ("Installing dependencies", self.install_dependencies),
            ("Downloading models", self.download_models),
            ("Initializing databases", self.init_databases),
            ("Creating configuration", self.create_config),
            ("Configuring Claude Desktop", self.configure_claude_desktop),
            ("Testing installation", self.test_installation),
        ]

        for step_name, step_func in steps:
            if not step_func():
                print(f"\n❌ Installation failed at: {step_name}")
                print("\nPlease resolve the error and run the installer again.")
                return False

        self.print_summary()
        return True


def main():
    """Main entry point"""
    installer = SecureRAGInstaller()

    try:
        success = installer.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⊘ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
