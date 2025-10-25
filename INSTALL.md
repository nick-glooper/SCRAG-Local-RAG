# SecureRAG v1.0 - Installation Instructions

## Quick Install (3 Steps)

### 1. Pull the Code

```bash
cd /path/to/SCRAG-Local-RAG

# Pull the SecureRAG v1.0 code
git pull origin claude/secure-rag-v1-build-011CUTkUWvPicwLS4zQZ27jB
```

That's it! All 30 files will be pulled.

### 2. Verify Files

```bash
ls -la
```

You should see:
- ✅ installer.py
- ✅ requirements.txt
- ✅ src/ folder with 13 Python files
- ✅ tests/ folder
- ✅ README.md (full guide)

### 3. Run Installer

```bash
python installer.py
```

The installer will:
- ✅ Check Python version
- ✅ Create virtual environment
- ✅ Install dependencies
- ✅ Download models (~2.5 GB)
- ✅ Configure Claude Desktop
- ✅ Test installation

---

## That's It!

After installation, restart Claude Desktop and you're ready to go.

## Need Help?

See [README.md](README.md) for complete documentation.
See [DEBUGGING.md](DEBUGGING.md) if you have issues.
