"""
AI Research Agent - Streamlit Application.

Main entry point for the evolutionary AI coding agent.
Connects all components: ingestion, research, and evolution.
"""

import asyncio
import streamlit as st
import traceback
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "evolution_log": [],
        "current_best": None,
        "research_status": "idle",  # idle, researching, evolving, complete, error
        "context_md": None,
        "tests_py": None,
        "generation_scores": [],
        "search_queries": [],
        "error_message": None,
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


init_session_state()


# =============================================================================
# Custom CSS
# =============================================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.8rem;
    }
    
    .status-idle { background: #2d3748; color: #a0aec0; }
    .status-researching { background: #2b6cb0; color: #bee3f8; }
    .status-evolving { background: #6b46c1; color: #d6bcfa; }
    .status-complete { background: #276749; color: #9ae6b4; }
    .status-error { background: #c53030; color: #feb2b2; }
    
    .evolution-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .score-display {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #48bb78 0%, #38a169 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    .search-query {
        background: rgba(59, 130, 246, 0.2);
        border-left: 3px solid #3b82f6;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Header
# =============================================================================

st.markdown('<h1 class="main-header">🧬 AI Research Agent</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #a0aec0; margin-bottom: 2rem;">'
    'Evolutionary coding agent powered by AlphaEvolve architecture • '
    'Parallel.ai DeepSearch • Poetiq async patterns'
    '</p>',
    unsafe_allow_html=True
)


# =============================================================================
# Sidebar - Inputs
# =============================================================================

with st.sidebar:
    st.header("📥 Input Configuration")
    
    # Problem description
    problem_description = st.text_area(
        "Problem Description",
        placeholder="Describe the coding problem you want to solve...\n\nExample: Write a Python function that finds the longest palindromic substring in a given string.",
        height=200,
    )
    
    st.divider()
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Documents (PDFs)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload reference documents to provide context",
    )
    
    # URL input
    reference_urls = st.text_area(
        "Reference URLs",
        placeholder="https://example.com/algorithm-docs\nhttps://another-reference.com",
        height=100,
        help="Enter URLs separated by newlines",
    )
    
    st.divider()
    
    # Evolution settings
    st.subheader("⚙️ Evolution Settings")
    
    num_generations = st.slider(
        "Max Generations",
        min_value=1,
        max_value=50,
        value=20,
        help="Maximum number of evolution iterations",
    )
    
    parallel_candidates = st.slider(
        "Parallel Candidates",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of code variants to generate per generation",
    )
    
    force_search = st.checkbox(
        "Force Deep Search",
        value=True,
        help="Always perform Parallel.ai search, even if context seems sufficient",
    )
    
    st.divider()
    
    # Action button
    start_button = st.button(
        "🚀 Start Research & Evolution",
        type="primary",
        use_container_width=True,
        disabled=not problem_description,
    )
    
    # Reset button
    if st.button("🔄 Reset", use_container_width=True):
        for key in ["evolution_log", "current_best", "context_md", "tests_py", 
                    "generation_scores", "search_queries", "error_message"]:
            st.session_state[key] = [] if "log" in key or "scores" in key or "queries" in key else None
        st.session_state.research_status = "idle"
        st.rerun()


# =============================================================================
# Main Content Area
# =============================================================================

# Status display
status_map = {
    "idle": ("status-idle", "Ready"),
    "researching": ("status-researching", "🔍 Researching..."),
    "evolving": ("status-evolving", "🧬 Evolving..."),
    "complete": ("status-complete", "✅ Complete"),
    "error": ("status-error", "❌ Error"),
}

status_class, status_text = status_map.get(
    st.session_state.research_status, 
    ("status-idle", "Ready")
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        f'<div style="text-align: center;">'
        f'<span class="status-badge {status_class}">{status_text}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

# Error display
if st.session_state.error_message:
    st.error(st.session_state.error_message)

st.divider()

# Main content tabs
tab_research, tab_evolution, tab_solution = st.tabs([
    "📚 Research", 
    "🧬 Evolution Log", 
    "💻 Best Solution"
])

with tab_research:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Research Context")
        
        if st.session_state.context_md:
            st.markdown(st.session_state.context_md)
        else:
            st.info("Research context will appear here after starting the agent.")
    
    with col2:
        st.subheader("Search Queries")
        if st.session_state.search_queries:
            for i, query in enumerate(st.session_state.search_queries, 1):
                st.markdown(
                    f'<div class="search-query">{i}. {query}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.caption("Search queries will appear here...")
    
    if st.session_state.tests_py:
        st.subheader("Generated Tests")
        with st.expander("View Test Code", expanded=False):
            st.code(st.session_state.tests_py, language="python")

with tab_evolution:
    st.subheader("Evolution Progress")
    
    # Metrics row
    if st.session_state.evolution_log:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Generation",
                len(st.session_state.generation_scores),
            )
        
        with col2:
            best_score = max(st.session_state.generation_scores) if st.session_state.generation_scores else 0
            st.metric(
                "Best Score",
                f"{best_score:.1%}",
            )
        
        with col3:
            st.metric(
                "Candidates",
                len(st.session_state.evolution_log),
            )
        
        with col4:
            if st.session_state.current_best:
                st.metric(
                    "Tests",
                    f"{st.session_state.current_best.get('tests_passed', 0)}/{st.session_state.current_best.get('tests_total', 0)}",
                )
    
    if st.session_state.generation_scores:
        # Score chart
        import pandas as pd
        chart_data = pd.DataFrame({
            "Generation": range(1, len(st.session_state.generation_scores) + 1),
            "Score": st.session_state.generation_scores,
        })
        st.line_chart(chart_data.set_index("Generation"), use_container_width=True)
    
    # Evolution log
    if st.session_state.evolution_log:
        st.subheader("Generation Details")
        for i, log_entry in enumerate(reversed(st.session_state.evolution_log[-10:])):
            gen_num = log_entry.get('generation', len(st.session_state.evolution_log) - i)
            score = log_entry.get('score', 0)
            status_icon = "✅" if score >= 1.0 else "🔄" if score > 0 else "❌"
            
            with st.expander(f"{status_icon} Generation {gen_num} - Score: {score:.1%}", expanded=(i == 0)):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Tests Passed:** {log_entry.get('tests_passed', 0)}/{log_entry.get('tests_total', 0)}")
                with col2:
                    st.markdown(f"**Status:** {log_entry.get('status', 'N/A')}")
                
                if log_entry.get("feedback"):
                    st.markdown("**Feedback:**")
                    st.code(log_entry["feedback"][:500], language="text")
                
                if log_entry.get("code"):
                    st.markdown("**Code:**")
                    code = log_entry.get("code", "")
                    st.code(code[:1000] + ("..." if len(code) > 1000 else ""), language="python")
    else:
        st.info("Evolution log will appear here as the agent iterates.")

with tab_solution:
    st.subheader("Best Solution")
    
    if st.session_state.current_best:
        best = st.session_state.current_best
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            score = best.get("score", 0)
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="score-display">{score:.1%}</div>'
                f'<div style="color: #a0aec0;">Test Pass Rate</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col2:
            st.metric("Generation", best.get("generation", "N/A"))
        with col3:
            st.metric("Tests", f"{best.get('tests_passed', 0)}/{best.get('tests_total', 0)}")
        
        st.divider()
        
        # Code display
        st.code(best.get("code", "# No solution yet"), language="python")
        
        # Download button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.download_button(
                label="📥 Download Solution",
                data=best.get("code", ""),
                file_name="solution.py",
                mime="text/x-python",
                use_container_width=True,
            )
    else:
        st.info("The best solution will appear here after evolution completes.")


# =============================================================================
# Pipeline Execution
# =============================================================================

async def run_evolution_pipeline(
    problem: str,
    urls: list[str],
    files: list,
    max_generations: int,
    num_candidates: int,
    force_search: bool,
):
    """
    Run the full research and evolution pipeline.
    """
    from agents.researcher import ResearcherAgent
    from agents.evolver import EvolverAgent, EvolutionState
    from utils.logger import start_run, get_logger
    
    # Start logging run
    run_name = f"evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = start_run(run_name)
    logger.set_config({
        "problem_preview": problem[:200],
        "max_generations": max_generations,
        "num_candidates": num_candidates,
        "force_search": force_search,
        "num_urls": len(urls) if urls else 0,
        "num_files": len(files) if files else 0,
    })
    logger.info("Pipeline started", phase="init")
    
    try:
        # Phase 1: Research
        st.session_state.research_status = "researching"
        logger.info("Research phase started", phase="research")
        
        researcher = ResearcherAgent()
        
        # Process uploaded files
        file_data = []
        if files:
            for f in files:
                file_data.append((f.name, f.read()))
        
        # Run research
        research_output = await researcher.research(
            problem=problem,
            urls=urls if urls else None,
            files=file_data if file_data else None,
            force_search=force_search,
        )
        
        st.session_state.context_md = research_output.context_md
        st.session_state.tests_py = research_output.tests_py
        st.session_state.search_queries = research_output.search_queries
        
        # Phase 2: Evolution
        st.session_state.research_status = "evolving"
        
        evolver = EvolverAgent(
            max_generations=max_generations,
            parallel_candidates=num_candidates,
        )
        
        def on_generation(state: EvolutionState):
            """Callback for each generation."""
            if state.best_solution:
                log_entry = {
                    "generation": state.generation,
                    "score": state.best_solution.score,
                    "tests_passed": state.best_solution.tests_passed,
                    "tests_total": state.best_solution.tests_total,
                    "status": "Improved" if state.generation > 0 else "Initial",
                    "feedback": state.best_solution.feedback,
                    "code": state.best_solution.code,
                    "timestamp": datetime.now().isoformat(),
                }
                
                st.session_state.evolution_log.append(log_entry)
                st.session_state.generation_scores.append(state.best_solution.score)
                
                # Update best
                if (st.session_state.current_best is None or 
                    state.best_solution.score > st.session_state.current_best.get("score", 0)):
                    st.session_state.current_best = log_entry
        
        result = await evolver.evolve(
            context=research_output.context_md,
            tests=research_output.tests_py,
            on_generation=on_generation,
        )
        
        # Final update
        if result.best_solution:
            st.session_state.current_best = {
                "generation": result.best_solution.generation,
                "score": result.best_solution.score,
                "tests_passed": result.best_solution.tests_passed,
                "tests_total": result.best_solution.tests_total,
                "code": result.best_solution.code,
                "feedback": result.best_solution.feedback,
            }
        
        st.session_state.research_status = "complete"
        
        # Finish logging
        logger.finish(
            status="success" if result.success else "failed",
            final_score=result.best_solution.score if result.best_solution else None,
            final_code=result.best_solution.code if result.best_solution else None,
        )
        logger.info(f"Pipeline complete. Log saved to: {logger.log_file}")
        
    except Exception as e:
        st.session_state.research_status = "error"
        # Capture full exception details for debugging
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else "(empty error message)"
        full_traceback = traceback.format_exc()
        detailed_error = f"{error_type}: {error_msg}\n\nTraceback:\n{full_traceback}"
        st.session_state.error_message = f"Pipeline error: {detailed_error}"
        logger.error(f"Pipeline failed: {error_type}: {error_msg}")
        logger.finish(status="failed")
        raise


if start_button:
    # Parse URLs
    urls = [u.strip() for u in reference_urls.split("\n") if u.strip()]
    
    # Reset state
    st.session_state.evolution_log = []
    st.session_state.generation_scores = []
    st.session_state.current_best = None
    st.session_state.context_md = None
    st.session_state.tests_py = None
    st.session_state.search_queries = []
    st.session_state.error_message = None
    
    # Run the pipeline
    with st.spinner("Running AI Research Agent..."):
        try:
            asyncio.run(run_evolution_pipeline(
                problem=problem_description,
                urls=urls,
                files=uploaded_files or [],
                max_generations=num_generations,
                num_candidates=parallel_candidates,
                force_search=force_search,
            ))
        except Exception as e:
            st.error(f"Error: {e}")
    
    st.rerun()


# =============================================================================
# Footer
# =============================================================================

st.divider()
st.markdown(
    '<p style="text-align: center; color: #4a5568; font-size: 0.8rem;">'
    '🧬 AI Research Agent • AlphaEvolve Architecture • '
    'Parallel.ai DeepSearch • Poetiq Async Patterns'
    '</p>',
    unsafe_allow_html=True
)
