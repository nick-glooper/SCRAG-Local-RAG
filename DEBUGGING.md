# SecureRAG Debugging Guide

This guide explains how to debug SecureRAG and access logs for troubleshooting.

## Log File Locations

SecureRAG logs everything to a single log file:

```
~/.secure-rag/logs/secure-rag.log
```

### What Gets Logged

- ✅ Component initialization
- ✅ Document ingestion progress
- ✅ Search queries and results
- ✅ Collection operations
- ✅ Errors and exceptions (with stack traces)
- ✅ Performance metrics
- ✅ Model loading status

## Quick Log Access

### Using the Log Viewer Script

We've included a convenient log viewer:

```bash
# Show last 50 lines
python scripts/view_logs.py

# Show last 100 lines
python scripts/view_logs.py --tail 100

# Show only errors and warnings
python scripts/view_logs.py --errors

# Follow log in real-time (like tail -f)
python scripts/view_logs.py --follow

# Show log file path and size
python scripts/view_logs.py --path

# Clear log file
python scripts/view_logs.py --clear
```

### Using Standard Unix Tools

```bash
# View entire log
cat ~/.secure-rag/logs/secure-rag.log

# View last 50 lines
tail -n 50 ~/.secure-rag/logs/secure-rag.log

# Follow log in real-time
tail -f ~/.secure-rag/logs/secure-rag.log

# Search for errors
grep -i error ~/.secure-rag/logs/secure-rag.log

# Search for specific term
grep -i "collection" ~/.secure-rag/logs/secure-rag.log

# View with pagination
less ~/.secure-rag/logs/secure-rag.log
```

## Log Format

Each log entry follows this format:

```
TIMESTAMP - LOGGER_NAME - LEVEL - MESSAGE
```

Example:
```
2024-03-15 10:23:45,123 - src.mcp_server - INFO - Searching 'Legal_Contracts' for: payment terms
2024-03-15 10:23:45,456 - src.vector_store - ERROR - Collection 'Invalid' not found
```

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General information about operations
- **WARNING**: Something unexpected but not critical
- **ERROR**: An error occurred but system continues
- **CRITICAL**: Severe error, system may not continue

## Common Debugging Scenarios

### Scenario 1: Installation Issues

**Problem**: Installation fails or components don't load

**How to Debug**:

```bash
# 1. Check if log file exists
ls -lh ~/.secure-rag/logs/secure-rag.log

# 2. View installation errors
python scripts/view_logs.py --errors

# 3. Check Python environment
which python
python --version

# 4. Verify dependencies
~/.secure-rag/venv/bin/pip list
```

**What to Look For**:
- Missing dependencies
- Version conflicts
- Permission errors
- Path issues

### Scenario 2: Document Ingestion Fails

**Problem**: PDF/DOCX ingestion fails or hangs

**How to Debug**:

```bash
# Follow log while ingesting
python scripts/view_logs.py --follow

# Then in Claude Desktop, try ingesting a document
# Watch the log output in real-time
```

**What to Look For**:
```
ERROR - Failed to extract PDF: [error details]
ERROR - No text could be extracted from document
ERROR - Embedding generation failed
ERROR - Collection 'XYZ' not found
```

### Scenario 3: Search Returns No Results

**Problem**: Searches complete but return no results

**How to Debug**:

```bash
# View recent log entries
python scripts/view_logs.py --tail 100

# Look for search operations
grep "Searching" ~/.secure-rag/logs/secure-rag.log
```

**What to Look For**:
```
INFO - Searching 'collection_name' for: query
INFO - Generated X chunks
INFO - Search returned 0 results  <-- This indicates the issue
```

**Common Causes**:
- Documents not properly ingested
- Wrong collection name
- Min confidence threshold too high
- Empty collection

### Scenario 4: Models Not Loading

**Problem**: Embedding or reranking models fail to load

**How to Debug**:

```bash
# Check for model loading errors
python scripts/view_logs.py --errors | grep -i "model"
```

**What to Look For**:
```
ERROR - Failed to load local model: [model_name]
ERROR - Model files not found
ERROR - Out of memory loading model
```

**Solutions**:
```bash
# Re-download models
~/.secure-rag/venv/bin/python scripts/download_models.py

# Check disk space
df -h

# Check memory
free -h  # Linux
vm_stat  # macOS
```

### Scenario 5: MCP Server Not Responding

**Problem**: Claude Desktop doesn't show SecureRAG tools

**How to Debug**:

```bash
# 1. Check if server starts
python scripts/view_logs.py --follow

# 2. Restart Claude Desktop and watch logs

# 3. Check Claude Desktop config
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json  # macOS
cat ~/.config/Claude/claude_desktop_config.json  # Linux
```

**What to Look For in Claude Desktop Logs**:
```
# macOS
~/Library/Logs/Claude/mcp*.log

# Linux
~/.config/Claude/logs/mcp*.log
```

## Extracting Logs for Support

When reporting issues, extract relevant log sections:

### Method 1: Copy Last N Lines

```bash
# Get last 100 lines
tail -n 100 ~/.secure-rag/logs/secure-rag.log > securerag_error.log

# Open in editor to copy
nano securerag_error.log
```

### Method 2: Extract Errors Only

```bash
# Get all errors from today
python scripts/view_logs.py --errors > securerag_errors.log

# Or with grep
grep -i error ~/.secure-rag/logs/secure-rag.log > securerag_errors.log
```

### Method 3: Extract Specific Operation

```bash
# Extract logs for a specific collection
grep "Legal_Contracts" ~/.secure-rag/logs/secure-rag.log > collection_debug.log

# Extract logs from a time range
grep "2024-03-15 14:" ~/.secure-rag/logs/secure-rag.log > afternoon_logs.log
```

## Advanced Debugging

### Enable DEBUG Logging

Edit `~/.secure-rag/config.yaml`:

```yaml
logging:
  level: DEBUG  # Change from INFO to DEBUG
  log_file: ~/.secure-rag/logs/secure-rag.log
```

⚠️ **Warning**: DEBUG logging is very verbose and will create large log files.

### Python Stack Traces

All Python exceptions are logged with full stack traces:

```python
ERROR - Error ingesting document: [error message]
Traceback (most recent call last):
  File "src/document_processor.py", line 123, in ingest_document
    ...
  [Full stack trace]
```

### Component-Specific Logging

Each component has its own logger:

```python
src.config           - Configuration loading
src.vector_store     - Vector database operations
src.embeddings       - Embedding generation
src.reranker         - Reranking operations
src.chunking         - Document chunking
src.document_processor - Document ingestion
src.collections      - Collection management
src.query_history    - Query logging
src.kb_mode          - KB mode state
src.versioning       - Version comparisons
src.backup           - Backup/restore
src.mcp_server       - MCP server and tools
```

Filter by component:

```bash
grep "src.embeddings" ~/.secure-rag/logs/secure-rag.log
```

## Performance Debugging

### Slow Operations

Look for timing information in logs:

```bash
grep "processing_time\|search_time" ~/.secure-rag/logs/secure-rag.log
```

Example output:
```
INFO - Ingestion complete: 847 chunks in 125.34s
INFO - Search complete: 5 results in 0.89s
```

### Memory Issues

```bash
# Check for memory-related errors
grep -i "memory\|oom" ~/.secure-rag/logs/secure-rag.log
```

If seeing memory issues:
1. Reduce batch size in config
2. Disable reranking temporarily
3. Use CPU instead of GPU (paradoxically uses less RAM)

## Log Rotation

If log file gets too large:

```bash
# Check size
ls -lh ~/.secure-rag/logs/secure-rag.log

# Archive old log
mv ~/.secure-rag/logs/secure-rag.log ~/.secure-rag/logs/secure-rag.log.old

# Or clear completely
python scripts/view_logs.py --clear
```

## Useful Commands Summary

```bash
# Quick error check
python scripts/view_logs.py --errors

# Follow in real-time
python scripts/view_logs.py --follow

# Get last 100 lines
python scripts/view_logs.py --tail 100

# Search for specific term
grep -i "search term" ~/.secure-rag/logs/secure-rag.log

# Count errors
grep -c ERROR ~/.secure-rag/logs/secure-rag.log

# Show errors with context (3 lines before/after)
grep -C 3 ERROR ~/.secure-rag/logs/secure-rag.log

# Export for support
tail -n 200 ~/.secure-rag/logs/secure-rag.log > debug_export.log
```

## Getting Help

When asking for help, always include:

1. **Log excerpt** showing the error:
   ```bash
   python scripts/view_logs.py --errors > error_log.txt
   ```

2. **System information**:
   ```bash
   python --version
   uname -a  # or `systeminfo` on Windows
   ```

3. **Configuration** (sanitize any API keys):
   ```bash
   cat ~/.secure-rag/config.yaml
   ```

4. **Steps to reproduce** the issue

5. **Expected vs actual behavior**

## Pro Tips

1. **Always check logs first** - Most issues are logged with helpful error messages

2. **Use `--follow` during operations** - Watch logs in real-time to see exactly where things fail

3. **Search for stack traces** - Python tracebacks show exactly where errors occur

4. **Check timestamps** - Ensure you're looking at recent log entries

5. **Clear old logs** - Start with a clean slate when debugging new issues

6. **Enable DEBUG temporarily** - More verbose logging can help isolate issues

7. **Compare with working state** - If something worked before, compare old vs new logs

---

**Need more help?** Open an issue on GitHub with your log excerpts.
