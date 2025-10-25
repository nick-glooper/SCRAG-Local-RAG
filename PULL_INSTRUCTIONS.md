# Instructions to Pull SecureRAG v1.0

## The Issue

The code is on branch: `claude/secure-rag-v1-build-011CUTkUWvPicwLS4zQZ27jB`

If you just did `git pull`, you probably pulled the `main` branch which only has the README.

## Solution - Run These Commands:

```bash
# 1. Navigate to your local repository
cd /path/to/SCRAG-Local-RAG

# 2. Fetch all branches from remote
git fetch origin

# 3. Check out the feature branch
git checkout claude/secure-rag-v1-build-011CUTkUWvPicwLS4zQZ27jB

# 4. Pull the latest changes
git pull origin claude/secure-rag-v1-build-011CUTkUWvPicwLS4zQZ27jB
```

## Verify You Have All Files

After checking out the branch, you should see:

```bash
ls -la
```

Expected output:
```
.gitignore
DEBUGGING.md
LICENSE
QUICK_DEBUG.md
README.md
config/
installer.py
requirements.txt
scripts/
setup.py
src/
tests/
```

## Check File Count

```bash
# Should show 22 Python files
find . -name "*.py" -type f | wc -l

# Should show all source files
ls src/
```

Expected in src/:
```
__init__.py
backup.py
chunking.py
collections.py
config.py
document_processor.py
embeddings.py
kb_mode.py
mcp_server.py
query_history.py
reranker.py
vector_store.py
versioning.py
```

## Alternative: Clone Fresh

If issues persist, clone the specific branch directly:

```bash
# Clone only the feature branch
git clone -b claude/secure-rag-v1-build-011CUTkUWvPicwLS4zQZ27jB \
  https://github.com/nick-glooper/SCRAG-Local-RAG.git SecureRAG-v1

cd SecureRAG-v1
ls -la
```

## Quick Verification Commands

```bash
# 1. Check which branch you're on
git branch

# Should show: * claude/secure-rag-v1-build-011CUTkUWvPicwLS4zQZ27jB

# 2. Check commit history
git log --oneline -3

# Should show:
# 47788a2 Add comprehensive debugging and logging tools
# 94acbac Complete SecureRAG v1.0 implementation
# 6e96365 Initial commit: Add README for SCRAG-Local-RAG

# 3. Verify all files
ls -R

# Should show config/, scripts/, src/, tests/ with many files
```

## If You Still Only See README

Run this diagnostic:

```bash
# Check current branch
git branch

# If it shows 'main' or 'master', that's the problem!
# You need to switch to the feature branch:
git checkout claude/secure-rag-v1-build-011CUTkUWvPicwLS4zQZ27jB
```

---

**TL;DR**: You're probably on the wrong branch. Run:
```bash
git fetch origin
git checkout claude/secure-rag-v1-build-011CUTkUWvPicwLS4zQZ27jB
```
