"""
SecureRAG - Local, Privacy-First RAG System for Claude Desktop
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

setup(
    name="secure-rag",
    version="1.0.0",
    description="Local, privacy-first RAG system for Claude Desktop via MCP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SecureRAG Team",
    author_email="contact@securerag.dev",
    url="https://github.com/secure-rag/secure-rag",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastmcp>=0.2.0",
        "qdrant-client>=1.7.0",
        "sentence-transformers>=2.2.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "pymupdf>=1.23.0",
        "python-docx>=0.8.11",
        "pypdf>=3.0.0",
        "langchain-text-splitters>=0.0.1",
        "cryptography>=41.0.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "cloud": [
            "openai>=1.0.0",
            "anthropic>=0.18.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    entry_points={
        "console_scripts": [
            "secure-rag=src.mcp_server:main",
        ],
    },
)
