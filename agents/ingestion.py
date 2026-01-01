"""
Content Ingestion Utilities.

Handles parsing and extracting content from various sources:
- URLs (web pages)
- PDF documents
- Text files
"""

import asyncio
import re
from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO

import aiohttp
from bs4 import BeautifulSoup

# Optional PDF support
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


@dataclass
class IngestedContent:
    """Structured content from ingestion."""
    source: str  # URL, filename, or identifier
    content: str  # Extracted text content
    content_type: str  # "url", "pdf", "text"
    title: str | None = None
    error: str | None = None


# =============================================================================
# URL Parsing
# =============================================================================

def extract_urls(text: str) -> list[str]:
    """
    Extract URLs from text using regex.
    
    Args:
        text: Input text that may contain URLs
        
    Returns:
        List of extracted URLs
    """
    # Regex pattern for URLs
    url_pattern = re.compile(
        r'https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)',  # path
        re.IGNORECASE
    )
    
    return url_pattern.findall(text)


async def fetch_url(
    url: str,
    timeout: float = 30.0,
) -> IngestedContent:
    """
    Fetch and parse content from a URL.
    
    Uses BeautifulSoup to extract text content from HTML.
    
    Args:
        url: The URL to fetch
        timeout: Request timeout in seconds
        
    Returns:
        IngestedContent with extracted text
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout),
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; AIResearchAgent/1.0)"
                },
            ) as response:
                if response.status != 200:
                    return IngestedContent(
                        source=url,
                        content="",
                        content_type="url",
                        error=f"HTTP {response.status}",
                    )
                
                html = await response.text()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                
                # Remove script and style elements
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()
                
                # Extract title
                title = soup.title.string if soup.title else None
                
                # Get text content
                text = soup.get_text(separator="\n", strip=True)
                
                # Clean up whitespace
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                clean_text = "\n".join(lines)
                
                return IngestedContent(
                    source=url,
                    content=clean_text[:50000],  # Limit content size
                    content_type="url",
                    title=title,
                )
                
    except asyncio.TimeoutError:
        return IngestedContent(
            source=url,
            content="",
            content_type="url",
            error="Request timed out",
        )
    except Exception as e:
        return IngestedContent(
            source=url,
            content="",
            content_type="url",
            error=str(e),
        )


async def fetch_urls(urls: list[str]) -> list[IngestedContent]:
    """
    Fetch multiple URLs concurrently.
    
    Args:
        urls: List of URLs to fetch
        
    Returns:
        List of IngestedContent in same order as input
    """
    tasks = [fetch_url(url) for url in urls]
    return await asyncio.gather(*tasks)


# =============================================================================
# PDF Parsing
# =============================================================================

def parse_pdf(file: BinaryIO | bytes, filename: str = "document.pdf") -> IngestedContent:
    """
    Extract text content from a PDF file.
    
    Args:
        file: File-like object or bytes of the PDF
        filename: Name of the file for reference
        
    Returns:
        IngestedContent with extracted text
    """
    if not HAS_PYPDF:
        return IngestedContent(
            source=filename,
            content="",
            content_type="pdf",
            error="pypdf not installed. Run: pip install pypdf",
        )
    
    try:
        # Handle bytes input
        if isinstance(file, bytes):
            file = BytesIO(file)
        
        reader = PdfReader(file)
        
        # Extract text from all pages
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_text.append(f"--- Page {i+1} ---\n{text}")
        
        full_text = "\n\n".join(pages_text)
        
        # Get title from metadata if available
        title = None
        if reader.metadata:
            title = reader.metadata.get("/Title")
        
        return IngestedContent(
            source=filename,
            content=full_text[:100000],  # Limit content size
            content_type="pdf",
            title=title,
        )
        
    except Exception as e:
        return IngestedContent(
            source=filename,
            content="",
            content_type="pdf",
            error=str(e),
        )


# =============================================================================
# Text File Parsing
# =============================================================================

def parse_text_file(
    file: BinaryIO | bytes | str,
    filename: str = "document.txt",
) -> IngestedContent:
    """
    Parse a text or markdown file.
    
    Args:
        file: File content (bytes, string, or file-like object)
        filename: Name of the file for reference
        
    Returns:
        IngestedContent with the text
    """
    try:
        if isinstance(file, bytes):
            content = file.decode("utf-8", errors="replace")
        elif isinstance(file, str):
            content = file
        else:
            content = file.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
        
        # Detect content type from extension
        if filename.endswith(".md"):
            content_type = "markdown"
        else:
            content_type = "text"
        
        return IngestedContent(
            source=filename,
            content=content[:100000],
            content_type=content_type,
        )
        
    except Exception as e:
        return IngestedContent(
            source=filename,
            content="",
            content_type="text",
            error=str(e),
        )


# =============================================================================
# Unified Ingestion
# =============================================================================

async def ingest_all(
    text_input: str,
    urls: list[str] | None = None,
    files: list[tuple[str, bytes]] | None = None,
) -> list[IngestedContent]:
    """
    Ingest content from all sources.
    
    Args:
        text_input: User's text input (may contain URLs)
        urls: Additional explicit URLs
        files: List of (filename, content) tuples
        
    Returns:
        List of all ingested content
    """
    results: list[IngestedContent] = []
    
    # Extract URLs from text
    extracted_urls = extract_urls(text_input)
    all_urls = list(set(extracted_urls + (urls or [])))
    
    # Fetch URLs concurrently
    if all_urls:
        url_results = await fetch_urls(all_urls)
        results.extend(url_results)
    
    # Process files
    for filename, content in (files or []):
        if filename.lower().endswith(".pdf"):
            result = parse_pdf(content, filename)
        else:
            result = parse_text_file(content, filename)
        results.append(result)
    
    return results


# =============================================================================
# Example usage
# =============================================================================

async def _demo():
    """Demo the ingestion functionality."""
    
    # Test URL extraction
    text = """
    Check out these resources:
    - https://docs.python.org/3/library/asyncio.html
    - https://example.com/guide
    Also see http://localhost:8000/api
    """
    
    urls = extract_urls(text)
    print(f"Extracted URLs: {urls}")
    
    # Test URL fetching
    print("\nFetching Python docs...")
    result = await fetch_url("https://docs.python.org/3/library/asyncio.html")
    print(f"Title: {result.title}")
    print(f"Content length: {len(result.content)} chars")
    print(f"First 200 chars: {result.content[:200]}")


if __name__ == "__main__":
    asyncio.run(_demo())
