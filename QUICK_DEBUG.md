# Quick Debug Reference

## 🚨 Problem? Start Here

```bash
# 1. Check for errors
python scripts/view_logs.py --errors

# 2. Follow log in real-time
python scripts/view_logs.py --follow

# 3. Check last 100 lines
python scripts/view_logs.py --tail 100
```

## 📍 Log File Location

```
~/.secure-rag/logs/secure-rag.log
```

## 🔍 Common Issues

### Installation Failed
```bash
# Check what went wrong
python scripts/view_logs.py --errors
cat ~/.secure-rag/logs/secure-rag.log
```

### Tools Not Showing in Claude Desktop
```bash
# 1. Check Claude Desktop config
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json  # macOS
cat ~/.config/Claude/claude_desktop_config.json  # Linux

# 2. Restart Claude Desktop completely

# 3. Check MCP logs
tail ~/Library/Logs/Claude/mcp*.log  # macOS
```

### Document Ingestion Fails
```bash
# Watch in real-time
python scripts/view_logs.py --follow

# Check specific errors
grep "ingest" ~/.secure-rag/logs/secure-rag.log | grep ERROR
```

### Search Returns Nothing
```bash
# Check if documents were ingested
grep "chunks_created" ~/.secure-rag/logs/secure-rag.log

# Check search operations
grep "Searching" ~/.secure-rag/logs/secure-rag.log
```

### Models Not Loading
```bash
# Re-download models
~/.secure-rag/venv/bin/python scripts/download_models.py

# Check model errors
grep -i "model" ~/.secure-rag/logs/secure-rag.log | grep ERROR
```

## 📤 Export Logs for Support

```bash
# Get recent errors
python scripts/view_logs.py --errors > my_errors.log

# Get last 200 lines
tail -n 200 ~/.secure-rag/logs/secure-rag.log > debug_export.log

# Get specific operation logs
grep "collection_name" ~/.secure-rag/logs/secure-rag.log > collection_debug.log
```

## 🛠️ Quick Fixes

### Clear Everything and Start Fresh
```bash
# Backup first!
cp -r ~/.secure-rag ~/.secure-rag.backup

# Clear logs
python scripts/view_logs.py --clear

# Re-run installer
python installer.py
```

### Enable More Verbose Logging
Edit `~/.secure-rag/config.yaml`:
```yaml
logging:
  level: DEBUG  # Change from INFO
```

### Check System Status
```bash
# Health check (from Claude)
"Run health check"

# Get stats (from Claude)
"Get system stats"
```

## 📚 Full Documentation

See [DEBUGGING.md](DEBUGGING.md) for comprehensive debugging guide.
