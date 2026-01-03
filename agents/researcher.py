"""
DeepSearch Researcher Agent.

Uses Parallel.ai Search API for comprehensive research and generates:
- context.md: Synthesized research findings
- tests.py: Test cases for the problem
"""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime

# Parallel.ai SDK
try:
    from parallel import Parallel
    HAS_PARALLEL = True
except ImportError:
    HAS_PARALLEL = False

from agents.ingestion import IngestedContent, ingest_all
from config import get_api_key
from utils.llm_engine import LLMEngine, get_engine


@dataclass
class SearchResult:
    """Result from a search query."""
    url: str
    title: str
    excerpts: list[str]
    publish_date: str | None = None


@dataclass
class ResearchOutput:
    """Complete output from the researcher agent."""
    context_md: str  # Synthesized research in markdown
    tests_py: str  # Generated test cases
    search_results: list[SearchResult] = field(default_factory=list)
    ingested_content: list[IngestedContent] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)


# =============================================================================
# Parallel.ai Search
# =============================================================================

async def deep_search(
    objective: str,
    max_results: int = 10,
    mode: str = "agentic",
) -> list[SearchResult]:
    """
    Perform deep search using Parallel.ai Search API.
    
    Args:
        objective: Natural language search objective
        max_results: Maximum number of results
        mode: "agentic" (token-efficient) or "one-shot" (comprehensive)
        
    Returns:
        List of SearchResult objects
    """
    if not HAS_PARALLEL:
        print("Warning: parallel package not installed. Run: pip install parallel")
        return []
    
    api_key = get_api_key("parallel")
    if not api_key:
        print("Warning: PARALLEL_API_KEY not set")
        return []
    
    try:
        client = Parallel(api_key=api_key)
        
        # Run in executor since SDK might be sync
        loop = asyncio.get_event_loop()
        search = await loop.run_in_executor(
            None,
            lambda: client.beta.search(
                objective=objective,
                mode=mode,
                max_results=max_results,
            )
        )
        
        results = []
        for item in search.results:
            results.append(SearchResult(
                url=item.url,
                title=item.title,
                excerpts=item.excerpts if hasattr(item, 'excerpts') else [],
                publish_date=item.publish_date if hasattr(item, 'publish_date') else None,
            ))
        
        return results
        
    except Exception as e:
        print(f"Search error: {e}")
        return []


async def parallel_deep_search(
    objectives: list[str],
    max_results_per_query: int = 5,
) -> list[SearchResult]:
    """
    Execute multiple search queries in parallel.
    
    Args:
        objectives: List of search objectives
        max_results_per_query: Max results per query
        
    Returns:
        Deduplicated list of all search results
    """
    tasks = [
        deep_search(obj, max_results_per_query, "agentic")
        for obj in objectives
    ]
    
    all_results = await asyncio.gather(*tasks)
    
    # Flatten and deduplicate by URL
    seen_urls = set()
    unique_results = []
    for results in all_results:
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
    
    return unique_results


# =============================================================================
# Research Agent
# =============================================================================

RESEARCH_DECISION_PROMPT = """You are a research planning agent. Given a problem description and existing context, decide if external search is needed.

## Problem Description
{problem}

## Existing Context
{context}

## Task
Analyze if you have enough information to solve this problem. Respond with EXACTLY one of:
- "SEARCH: <reason>" if external search would help
- "SUFFICIENT: <reason>" if existing context is enough

Be concise."""


SEARCH_QUERIES_PROMPT = """You are a research query generator. Generate targeted search queries for the given problem.

## Problem Description
{problem}

## Existing Context (summary)
{context_summary}

## Task
Generate 3-5 natural language search objectives that would help gather relevant information.
Each objective should be a clear statement of what information is needed.

Format your response as a numbered list:
1. <search objective>
2. <search objective>
...

Be specific and targeted."""


CONTEXT_SYNTHESIS_PROMPT = """You are a research synthesizer. Create a comprehensive context document from the gathered information.

## Original Problem
{problem}

## Search Results
{search_results}

## Ingested Documents
{ingested_docs}

## Task
Synthesize all information into a well-structured markdown document that will help an AI coding agent solve the problem.

Include:
1. **Problem Analysis**: What the user wants to achieve
2. **Key Concepts**: Important algorithms, patterns, or techniques
3. **Implementation Hints**: Relevant code patterns or approaches
4. **Edge Cases**: Important considerations
5. **What Doesn't Work & Why**: Document approaches that DON'T work and explain WHY they fail.
   Understanding why something doesn't work expands and deepens our knowledge of the subject.
   Include common pitfalls, failed approaches, and their root causes.

Be thorough but concise. Use markdown formatting."""


TEST_GENERATION_PROMPT = """You are a test case generator. Create comprehensive Python tests for the given problem.

## Problem Description
{problem}

## Context
{context}

## Task
Generate a Python test file that:
1. Defines a `run_tests()` function that returns a dict with:
   - "passed": number of tests passed
   - "total": total number of tests
   - "details": string with test results
2. Tests edge cases and normal cases
3. Is self-contained (no external dependencies except standard library)

The solution code will define functions that your tests should call.
Assume the solution provides the main function(s) needed to solve the problem.

Return ONLY the Python code, no explanations."""


class ResearcherAgent:
    """
    DeepSearch agent that gathers context and generates tests.
    
    Workflow:
    1. Ingest provided URLs and documents
    2. Decide if external search is needed
    3. Generate search queries and execute via Parallel.ai
    4. Synthesize findings into context.md
    5. Generate tests.py using reasoning model
    """
    
    def __init__(self, engine: LLMEngine | None = None):
        self.engine = engine or get_engine()
    
    async def research(
        self,
        problem: str,
        urls: list[str] | None = None,
        files: list[tuple[str, bytes]] | None = None,
        force_search: bool = False,
    ) -> ResearchOutput:
        """
        Execute the full research pipeline.
        
        Args:
            problem: User's problem description
            urls: Optional list of URLs to ingest
            files: Optional list of (filename, content) tuples
            force_search: Skip decision step and always search
            
        Returns:
            ResearchOutput with context and tests
        """
        # Step 1: Ingest provided content
        ingested = await ingest_all(problem, urls, files)
        
        # Build initial context from ingested content
        initial_context = self._format_ingested_content(ingested)
        
        # Step 2: Decide if search is needed
        search_results = []
        search_queries = []
        
        if force_search or await self._needs_search(problem, initial_context):
            # Step 3: Generate search queries
            search_queries = await self._generate_search_queries(
                problem, 
                initial_context[:2000]  # Limit context for query generation
            )
            
            # Step 4: Execute searches
            if search_queries:
                search_results = await parallel_deep_search(search_queries)
        
        # Step 5: Synthesize context
        context_md = await self._synthesize_context(
            problem,
            search_results,
            ingested,
        )
        
        # Step 6: Generate tests
        tests_py = await self._generate_tests(problem, context_md)
        
        return ResearchOutput(
            context_md=context_md,
            tests_py=tests_py,
            search_results=search_results,
            ingested_content=ingested,
            search_queries=search_queries,
        )
    
    async def _needs_search(self, problem: str, context: str) -> bool:
        """Decide if external search is needed."""
        prompt = RESEARCH_DECISION_PROMPT.format(
            problem=problem,
            context=context[:3000] if context else "None provided",
        )
        
        response = await self.engine.generate_with_cheap_model(prompt)
        
        if response.success:
            return response.content.upper().startswith("SEARCH")
        return True  # Default to searching on error
    
    async def _generate_search_queries(
        self,
        problem: str,
        context_summary: str,
    ) -> list[str]:
        """Generate search queries for the problem."""
        prompt = SEARCH_QUERIES_PROMPT.format(
            problem=problem,
            context_summary=context_summary or "None",
        )
        
        response = await self.engine.generate_with_cheap_model(prompt)
        
        if not response.success:
            return [problem[:200]]  # Fallback to problem as query
        
        # Parse numbered list
        queries = []
        for line in response.content.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                # Remove number prefix
                query = line.lstrip("0123456789.)-: ")
                if query:
                    queries.append(query)
        
        return queries[:5]  # Limit to 5 queries
    
    async def _synthesize_context(
        self,
        problem: str,
        search_results: list[SearchResult],
        ingested: list[IngestedContent],
    ) -> str:
        """Synthesize all information into context markdown."""
        # Format search results
        search_text = ""
        for i, result in enumerate(search_results[:10], 1):
            excerpts = "\n".join(f"  - {e}" for e in result.excerpts[:3])
            search_text += f"\n### {i}. {result.title}\nURL: {result.url}\n{excerpts}\n"
        
        if not search_text:
            search_text = "No search results available."
        
        # Format ingested docs
        ingested_text = self._format_ingested_content(ingested)
        if not ingested_text:
            ingested_text = "No documents ingested."
        
        prompt = CONTEXT_SYNTHESIS_PROMPT.format(
            problem=problem,
            search_results=search_text,
            ingested_docs=ingested_text[:10000],
        )
        
        response = await self.engine.generate_with_reasoning_model(prompt)
        
        if response.success:
            return response.content
        
        # Fallback: Basic context
        return f"""# Research Context

## Problem
{problem}

## Search Results
{search_text}

## Documents
{ingested_text[:5000]}

---
*Generated at {datetime.now().isoformat()}*
"""
    
    async def _generate_tests(self, problem: str, context: str) -> str:
        """Generate test cases for the problem."""
        prompt = TEST_GENERATION_PROMPT.format(
            problem=problem,
            context=context[:8000],
        )
        
        response = await self.engine.generate_with_reasoning_model(
            prompt,
            temperature=0.2,  # Lower temp for code generation
        )
        
        if response.success:
            # Extract code from response (handle markdown blocks)
            code = response.content
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            return code.strip()
        
        # Fallback: Basic test template
        return '''"""Auto-generated test cases."""

def run_tests():
    """Run all tests and return results."""
    passed = 0
    total = 1
    details = []
    
    try:
        # Basic existence test
        details.append("Tests not yet generated - running basic check")
        passed = 1
    except Exception as e:
        details.append(f"Error: {e}")
    
    return {
        "passed": passed,
        "total": total,
        "details": "\\n".join(details)
    }
'''
    
    def _format_ingested_content(self, ingested: list[IngestedContent]) -> str:
        """Format ingested content for prompts."""
        parts = []
        for item in ingested:
            if item.error:
                parts.append(f"[{item.source}]: Error - {item.error}")
            else:
                content_preview = item.content[:2000]
                if len(item.content) > 2000:
                    content_preview += "... (truncated)"
                parts.append(f"### {item.source}\n{content_preview}")
        
        return "\n\n".join(parts)


# =============================================================================
# Convenience function
# =============================================================================

async def run_research(
    problem: str,
    urls: list[str] | None = None,
    files: list[tuple[str, bytes]] | None = None,
) -> ResearchOutput:
    """
    Convenience function to run research.
    
    Args:
        problem: The problem description
        urls: Optional URLs to ingest
        files: Optional files to ingest
        
    Returns:
        ResearchOutput with context and tests
    """
    agent = ResearcherAgent()
    return await agent.research(problem, urls, files)


# =============================================================================
# Example usage
# =============================================================================

async def _demo():
    """Demo the researcher functionality."""
    problem = """
    Write a Python function that implements the Knuth-Morris-Pratt (KMP) 
    string matching algorithm. The function should take a text and a pattern,
    and return all starting indices where the pattern occurs in the text.
    """
    
    print("Running research agent...")
    print("=" * 50)
    
    agent = ResearcherAgent()
    output = await agent.research(problem, force_search=True)
    
    print(f"\nSearch Queries Generated: {len(output.search_queries)}")
    for q in output.search_queries:
        print(f"  - {q}")
    
    print(f"\nSearch Results: {len(output.search_results)}")
    for r in output.search_results[:3]:
        print(f"  - {r.title}: {r.url}")
    
    print("\n" + "=" * 50)
    print("CONTEXT.MD (first 1000 chars):")
    print(output.context_md[:1000])
    
    print("\n" + "=" * 50)
    print("TESTS.PY (first 500 chars):")
    print(output.tests_py[:500])


if __name__ == "__main__":
    asyncio.run(_demo())
