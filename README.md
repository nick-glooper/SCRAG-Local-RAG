# SecureRAG v1.0

> **DEPRECATED**: This project has been superseded by the unified [kbase-manager](https://github.com/nick-glooper/kbase-manager) system.
>
> The kbase-manager project provides:
> - **KB Manager GUI** - Desktop app for knowledge base management
> - **kb-mcp** - Unified MCP server for documents and conversations
> - **kb-cli** - Command-line interface for automation
> - **Systemd integration** - Automatic background sync
>
> All new development is happening in kbase-manager. This repository is archived for reference only.

---

**Local, Privacy-First RAG System for Claude Desktop**

SecureRAG brings NotebookLM-like functionality to Claude Desktop via the Model Context Protocol (MCP). Perfect for professionals who cannot use cloud services: government contractors, legal professionals, healthcare workers, and financial analysts.

## 🔒 Privacy First

- ✅ **100% Local** - All data remains on your machine
- ✅ **Zero Cloud Dependencies** - Works completely offline (except optional API keys)
- ✅ **Encrypted Backups** - AES-256 encryption for exports
- ✅ **Audit Trail** - Complete query history tracking
- ✅ **Open Source** - Full transparency, no telemetry

## ✨ Features

### Core Capabilities

- **Multi-Format Support**: PDF, DOCX, TXT, MD
- **Semantic Chunking**: Intelligent document splitting that respects structure
- **Advanced Search**: Vector search with optional reranking for accuracy
- **Document Versioning**: Track changes across document versions
- **Collection Management**: Organize documents into topic-based collections
- **KB-Only Mode**: Restrict Claude to only your knowledge base
- **Encrypted Backups**: Export and import collections securely

### Technical Highlights

- **Local Embeddings**: BAAI/bge-large-en-v1.5 (1024 dimensions)
- **Reranking**: BAAI/bge-reranker-large for improved accuracy
- **Vector Database**: Qdrant (embedded mode, no server needed)
- **MCP Integration**: Native Claude Desktop integration via FastMCP
- **Metadata Filtering**: Filter by document properties, dates, custom tags
- **Query History**: SQLite-based search tracking

## 📋 Requirements

- **Python**: 3.10 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: ~5GB for models + your document storage
- **OS**: macOS, Windows, or Linux
- **Claude Desktop**: Latest version with MCP support

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/SecureRAG.git
cd SecureRAG
```

### 2. Run Installer

```bash
python installer.py
```

The installer will:
- ✅ Check Python version
- ✅ Create virtual environment
- ✅ Install dependencies
- ✅ Download ML models (~2.5 GB)
- ✅ Initialize databases
- ✅ Configure Claude Desktop
- ✅ Test installation

### 3. Restart Claude Desktop

After installation completes, **restart Claude Desktop** for changes to take effect.

### 4. Start Using SecureRAG

Open Claude Desktop and try:

```
Create a collection called "Legal_Contracts"
```

```
Ingest /path/to/contract.pdf into Legal_Contracts with metadata:
client=Acme Corp, date=2024-03-15, type=NDA
```

```
Search Legal_Contracts for "termination clauses"
```

## 📚 Usage Guide

### Creating Collections

Collections organize your documents by topic, project, or client.

```
Create a collection called IBM_Contracts with description:
"All IBM-related contracts and agreements"
```

### Ingesting Documents

Add documents to your collections:

```
Ingest /Users/john/Documents/IBM_Master_Agreement.pdf into IBM_Contracts
with metadata: client=IBM, date=2024-01-15, type=Master Agreement,
status=Active, value=1000000
```

Supported formats: PDF, DOCX, TXT, MD

### Searching Your Knowledge Base

Basic search:
```
Search IBM_Contracts for "payment terms"
```

Advanced search with filters:
```
Search IBM_Contracts for "liability clauses" with filters:
type=Master Agreement, status=Active
```

Get more results:
```
Search IBM_Contracts for "force majeure" with top_k=10
```

### KB-Only Mode

Restrict Claude to ONLY use your knowledge base:

```
Enable KB mode for IBM_Contracts
```

Now Claude will only answer from your documents. Disable with:

```
Disable KB mode
```

### Document Versioning

Track changes across document versions:

```
Ingest /path/to/contract_v2.pdf into IBM_Contracts
with document_id=IBM_MSA version=v2
```

Compare versions:
```
Compare versions v1 and v2 of document IBM_MSA in IBM_Contracts
```

### Backup & Restore

Export a collection:
```
Export IBM_Contracts to /path/to/backup.tar.gz with password: mypassword
```

Import a collection:
```
Import /path/to/backup.tar.gz with password: mypassword as IBM_Contracts_Restored
```

### Collection Management

List all collections:
```
List all collections
```

Get detailed info:
```
Get info for collection IBM_Contracts
```

Delete a collection:
```
Delete collection IBM_Contracts (confirm: true)
```

### Query History

View your search history:
```
Show query history for IBM_Contracts (limit: 20)
```

### System Stats

Check system status:
```
Get system stats
```

```
Health check
```

## 🛠️ Configuration

Configuration file: `~/.secure-rag/config.yaml`

### Key Settings

```yaml
embeddings:
  provider: local  # local, openai, anthropic
  model: BAAI/bge-large-en-v1.5
  device: auto  # auto, cpu, cuda, mps

reranking:
  enabled: true
  model: BAAI/bge-reranker-large
  top_n_rerank: 20

chunking:
  strategy: semantic
  min_chunk_size: 256
  max_chunk_size: 1024
  chunk_overlap: 100

search:
  default_top_k: 5
  min_confidence: 0.0

security:
  encryption_enabled: true
  backup_encryption: true
```

### Using Cloud Embeddings (Optional)

To use OpenAI embeddings:

```yaml
embeddings:
  provider: openai
  model: text-embedding-3-large
  openai_api_key: ${OPENAI_API_KEY}
```

Set environment variable:
```bash
export OPENAI_API_KEY="your-api-key"
```

⚠️ **Warning**: Cloud embeddings send your data to external services.

## 📁 Directory Structure

```
~/.secure-rag/
├── config.yaml           # Configuration file
├── vector_db/            # Qdrant vector database
├── backups/              # Exported collections
├── models/               # Downloaded ML models
├── logs/                 # Application logs
├── query_history.db      # Search history (SQLite)
└── venv/                 # Python virtual environment
```

## 🔧 Advanced Usage

### Custom Metadata

Attach rich metadata for advanced filtering:

```
Ingest contract.pdf into Legal with metadata:
{
  "client": "Acme Corp",
  "date": "2024-03-15",
  "type": "NDA",
  "department": "Engineering",
  "value": 500000,
  "expires": "2025-03-15",
  "parties": ["Acme Corp", "Our Company"],
  "jurisdiction": "Delaware",
  "status": "Active"
}
```

Search with filters:
```
Search Legal for "confidentiality obligations" with filters:
{
  "client": "Acme Corp",
  "status": "Active",
  "type": "NDA"
}
```

### Performance Tuning

Edit `~/.secure-rag/config.yaml`:

```yaml
embeddings:
  batch_size: 64  # Increase for faster processing (if RAM allows)
  device: cuda    # Use GPU if available

reranking:
  enabled: true   # Better accuracy, slower
  top_n_rerank: 30  # Rerank more candidates

chunking:
  max_chunk_size: 512  # Smaller chunks = more precise, more storage
```

## 🧪 Testing

Run tests:

```bash
cd SecureRAG
source ~/.secure-rag/venv/bin/activate  # On Windows: .\.secure-rag\venv\Scripts\activate
pytest tests/
```

## 🐛 Troubleshooting

### Models Not Found

Re-download models:
```bash
~/.secure-rag/venv/bin/python scripts/download_models.py
```

### Claude Desktop Not Showing Tools

1. Check configuration:
   ```bash
   cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

2. Verify `secure-rag` entry exists

3. Restart Claude Desktop completely

4. Check logs:
   ```bash
   tail -f ~/.secure-rag/logs/secure-rag.log
   ```

### Out of Memory

Reduce batch size in config:
```yaml
embeddings:
  batch_size: 16
```

### Slow Search

Enable GPU if available:
```yaml
embeddings:
  device: cuda  # or mps for Apple Silicon
reranking:
  device: cuda
```

### Import Errors

Reinstall dependencies:
```bash
~/.secure-rag/venv/bin/pip install -r requirements.txt --force-reinstall
```

## 📊 Performance Expectations

- **Ingestion**: 100-page PDF in ~2 minutes
- **Search**: < 1 second per query
- **Embedding**: ~10 seconds per 100 chunks
- **Reranking**: ~500ms for 20 candidates

## 🔐 Security

### Data Privacy

- All embeddings computed locally
- No data sent to external services (unless using cloud embeddings)
- All storage is local
- Encrypted backups with AES-256
- Query history stored locally in SQLite

### Backup Encryption

Backups use PBKDF2 for key derivation (100,000 iterations) and Fernet (AES-128) for encryption.

To change iterations:
```yaml
security:
  pbkdf2_iterations: 200000  # More secure, slower
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/SecureRAG/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/SecureRAG/discussions)

## 🙏 Acknowledgments

- **FastMCP**: MCP server framework
- **Qdrant**: Vector database
- **BGE Models**: Embedding and reranking models from BAAI
- **LangChain**: Text splitting utilities
- **PyMuPDF**: PDF processing

## 🗺️ Roadmap

### v1.1 (Planned)

- [ ] OCR support for scanned PDFs
- [ ] Excel/CSV support
- [ ] Web page ingestion
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Collaborative collections

### v2.0 (Future)

- [ ] Real-time document monitoring
- [ ] Automated summarization
- [ ] Entity extraction
- [ ] Relationship graphs
- [ ] Custom fine-tuning

## 📈 Changelog

### v1.0.0 (2024-03-15)

- Initial release
- 20 MCP tools for document management and search
- Local embeddings and reranking
- Document versioning
- Encrypted backups
- KB-only mode
- Query history
- Comprehensive metadata support

---

**Built with ❤️ for privacy-conscious professionals**
