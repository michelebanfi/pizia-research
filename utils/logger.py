"""
Structured Logging for AI Research Agent.

Provides JSON-based logging with run tracking for:
- Debugging during development
- Post-run analysis by the researcher
- Tracking LLM calls, costs, and evolution progress

Also saves all artifacts (inputs, generated code, test results) to a run folder.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

# Log directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Runs directory (for artifacts)
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)


LogLevel = Literal["DEBUG", "INFO", "WARN", "ERROR"]


class RunLogger:
    """
    Structured logger for a single run.
    
    Creates a JSON log file per run with all events, LLM calls, and results.
    Also creates a folder with all artifacts (inputs, iterations, results).
    """
    
    def __init__(self, run_name: str | None = None):
        """
        Initialize a new run logger.
        
        Args:
            run_name: Optional name for the run. Defaults to timestamp.
        """
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"evolution_{self.run_id}"
        self.log_file = LOGS_DIR / f"{self.run_name}.json"
        
        # Create run artifacts folder
        self.run_dir = RUNS_DIR / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subfolders
        self.inputs_dir = self.run_dir / "inputs"
        self.inputs_dir.mkdir(exist_ok=True)
        
        self.iterations_dir = self.run_dir / "iterations"
        self.iterations_dir.mkdir(exist_ok=True)
        
        self.results_dir = self.run_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize log structure
        self.log_data = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "started_at": datetime.now().isoformat(),
            "ended_at": None,
            "status": "running",
            "config": {},
            "events": [],
            "llm_calls": [],
            "evolution_history": [],
            "summary": {},
            "artifacts_dir": str(self.run_dir),
        }
        
        self._save()
    
    def _save(self):
        """Save log to file."""
        with open(self.log_file, "w") as f:
            json.dump(self.log_data, f, indent=2, default=str)
    
    def _timestamp(self) -> str:
        return datetime.now().isoformat()
    
    def _save_artifact(self, subdir: Path, filename: str, content: str) -> Path:
        """Save an artifact file and return its path."""
        filepath = subdir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath
    
    # =========================================================================
    # Artifact Saving Methods
    # =========================================================================
    
    def save_input_problem(self, problem: str):
        """Save the input problem description."""
        self._save_artifact(self.inputs_dir, "problem.txt", problem)
        self.debug(f"Saved input problem to {self.inputs_dir / 'problem.txt'}")
    
    def save_input_urls(self, urls: list[str]):
        """Save the input URLs."""
        content = "\n".join(urls)
        self._save_artifact(self.inputs_dir, "urls.txt", content)
        self.debug(f"Saved {len(urls)} URLs to {self.inputs_dir / 'urls.txt'}")
    
    def save_input_file(self, filename: str, content: str | bytes):
        """Save an uploaded input file."""
        if isinstance(content, bytes):
            filepath = self.inputs_dir / filename
            with open(filepath, "wb") as f:
                f.write(content)
        else:
            self._save_artifact(self.inputs_dir, filename, content)
        self.debug(f"Saved input file: {filename}")
    
    def save_research_context(self, context_md: str):
        """Save the synthesized research context."""
        self._save_artifact(self.inputs_dir, "research_context.md", context_md)
        self.debug("Saved research context")
    
    def save_search_queries(self, queries: list[str]):
        """Save the generated search queries."""
        content = "\n".join(f"- {q}" for q in queries)
        self._save_artifact(self.inputs_dir, "search_queries.txt", content)
        self.debug(f"Saved {len(queries)} search queries")
    
    def save_tests(self, tests_py: str):
        """Save the generated test cases."""
        self._save_artifact(self.inputs_dir, "tests.py", tests_py)
        self.debug("Saved test cases")
    
    def save_iteration(
        self,
        generation: int,
        candidate_id: int,
        code: str,
        score: float,
        tests_passed: int,
        tests_total: int,
        feedback: str,
        is_best: bool = False,
    ):
        """
        Save a single iteration/candidate result.
        
        Args:
            generation: Generation number (1-indexed)
            candidate_id: Candidate number within generation (1-indexed)
            code: The generated code
            score: Test score (0.0 to 1.0)
            tests_passed: Number of tests passed
            tests_total: Total number of tests
            feedback: Execution feedback/errors
            is_best: Whether this is the best candidate of the generation
        """
        gen_dir = self.iterations_dir / f"gen_{generation:02d}"
        gen_dir.mkdir(exist_ok=True)
        
        # Save the code
        code_filename = f"candidate_{candidate_id:02d}.py"
        if is_best:
            code_filename = f"candidate_{candidate_id:02d}_BEST.py"
        self._save_artifact(gen_dir, code_filename, code)
        
        # Save the results
        results = {
            "generation": generation,
            "candidate_id": candidate_id,
            "score": score,
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "is_best": is_best,
            "feedback": feedback,
            "timestamp": self._timestamp(),
        }
        results_filename = f"candidate_{candidate_id:02d}_results.json"
        self._save_artifact(gen_dir, results_filename, json.dumps(results, indent=2))
        
        self.debug(f"Saved iteration gen={generation} candidate={candidate_id} score={score:.2%}")
    
    def save_generation_summary(
        self,
        generation: int,
        best_score: float,
        all_scores: list[float],
        best_code: str,
    ):
        """Save a summary of the generation."""
        gen_dir = self.iterations_dir / f"gen_{generation:02d}"
        gen_dir.mkdir(exist_ok=True)
        
        summary = {
            "generation": generation,
            "best_score": best_score,
            "avg_score": sum(all_scores) / len(all_scores) if all_scores else 0,
            "all_scores": all_scores,
            "candidates_count": len(all_scores),
            "timestamp": self._timestamp(),
        }
        self._save_artifact(gen_dir, "summary.json", json.dumps(summary, indent=2))
        
        # Also save best code at top level of generation
        self._save_artifact(gen_dir, "best.py", best_code)
    
    def save_final_result(
        self,
        success: bool,
        final_score: float | None,
        final_code: str | None,
        total_generations: int,
    ):
        """Save the final result of the evolution."""
        # Save final code
        if final_code:
            self._save_artifact(self.results_dir, "final_solution.py", final_code)
        
        # Save summary
        summary = {
            "success": success,
            "final_score": final_score,
            "total_generations": total_generations,
            "timestamp": self._timestamp(),
            "run_id": self.run_id,
        }
        self._save_artifact(self.results_dir, "summary.json", json.dumps(summary, indent=2))
        
        self.info(f"Final results saved to {self.results_dir}")
    
    # =========================================================================
    # Original Logging Methods
    # =========================================================================
    
    # =========================================================================
    # Original Logging Methods
    # =========================================================================
    
    def set_config(self, config: dict[str, Any]):
        """Record run configuration."""
        self.log_data["config"] = config
        self._save()
    
    def log(self, level: LogLevel, message: str, **data):
        """
        Log a general event.
        
        Args:
            level: Log level (DEBUG, INFO, WARN, ERROR)
            message: Human-readable message
            **data: Additional structured data
        """
        event = {
            "timestamp": self._timestamp(),
            "level": level,
            "message": message,
            **data,
        }
        self.log_data["events"].append(event)
        
        # Also print to console
        prefix = {"DEBUG": "🔍", "INFO": "ℹ️", "WARN": "⚠️", "ERROR": "❌"}.get(level, "")
        print(f"{prefix} [{level}] {message}")
        
        self._save()
    
    def debug(self, message: str, **data):
        self.log("DEBUG", message, **data)
    
    def info(self, message: str, **data):
        self.log("INFO", message, **data)
    
    def warn(self, message: str, **data):
        self.log("WARN", message, **data)
    
    def error(self, message: str, **data):
        self.log("ERROR", message, **data)
    
    def log_llm_call(
        self,
        model: str,
        prompt_preview: str,
        response_preview: str,
        duration: float,
        prompt_tokens: int,
        completion_tokens: int,
        success: bool,
        error: str | None = None,
        purpose: str = "general",
    ):
        """
        Log an LLM call with full details.
        
        Args:
            model: Model name used
            prompt_preview: First N chars of prompt
            response_preview: First N chars of response
            duration: Call duration in seconds
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            success: Whether call succeeded
            error: Error message if failed
            purpose: What this call was for (e.g., "code_generation", "test_generation")
        """
        call = {
            "timestamp": self._timestamp(),
            "model": model,
            "purpose": purpose,
            "prompt_preview": prompt_preview[:500] + "..." if len(prompt_preview) > 500 else prompt_preview,
            "response_preview": response_preview[:500] + "..." if len(response_preview) > 500 else response_preview,
            "duration_sec": round(duration, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "success": success,
            "error": error,
        }
        self.log_data["llm_calls"].append(call)
        self._save()
    
    def log_generation(
        self,
        generation: int,
        candidates_count: int,
        best_score: float,
        best_code_preview: str,
        all_scores: list[float],
    ):
        """
        Log an evolution generation.
        
        Args:
            generation: Generation number
            candidates_count: Number of candidates evaluated
            best_score: Best score achieved
            best_code_preview: Preview of best code
            all_scores: All candidate scores
        """
        gen_data = {
            "timestamp": self._timestamp(),
            "generation": generation,
            "candidates_count": candidates_count,
            "best_score": best_score,
            "avg_score": sum(all_scores) / len(all_scores) if all_scores else 0,
            "all_scores": all_scores,
            "best_code_preview": best_code_preview[:300] + "..." if len(best_code_preview) > 300 else best_code_preview,
        }
        self.log_data["evolution_history"].append(gen_data)
        
        self.info(
            f"Generation {generation}: best={best_score:.2f}, avg={gen_data['avg_score']:.2f}",
            generation=generation,
        )
        self._save()
    
    def finish(
        self,
        status: Literal["success", "failed", "cancelled"] = "success",
        final_score: float | None = None,
        final_code: str | None = None,
    ):
        """
        Mark the run as finished.
        
        Args:
            status: Final status
            final_score: Final best score
            final_code: Final best code
        """
        self.log_data["ended_at"] = self._timestamp()
        self.log_data["status"] = status
        
        # Calculate summary
        llm_calls = self.log_data["llm_calls"]
        total_tokens = sum(c["total_tokens"] for c in llm_calls)
        total_duration = sum(c["duration_sec"] for c in llm_calls)
        
        self.log_data["summary"] = {
            "total_llm_calls": len(llm_calls),
            "total_tokens": total_tokens,
            "total_llm_duration_sec": round(total_duration, 2),
            "total_generations": len(self.log_data["evolution_history"]),
            "final_score": final_score,
            "final_code_preview": final_code[:500] + "..." if final_code and len(final_code) > 500 else final_code,
        }
        
        self.info(
            f"Run finished: {status}",
            final_score=final_score,
            total_tokens=total_tokens,
        )
        self._save()
        
        return self.log_file


# Global logger instance for the current run
_current_logger: RunLogger | None = None


def start_run(run_name: str | None = None) -> RunLogger:
    """Start a new logging run."""
    global _current_logger
    _current_logger = RunLogger(run_name)
    return _current_logger


def get_logger() -> RunLogger | None:
    """Get the current run logger."""
    return _current_logger


def get_or_create_logger() -> RunLogger:
    """Get existing logger or create one."""
    global _current_logger
    if _current_logger is None:
        _current_logger = RunLogger()
    return _current_logger


# =============================================================================
# Log Analysis Utilities
# =============================================================================

def list_runs() -> list[Path]:
    """List all log files."""
    return sorted(LOGS_DIR.glob("*.json"), reverse=True)


def load_run(log_file: Path | str) -> dict:
    """Load a run log from file."""
    with open(log_file) as f:
        return json.load(f)


def get_latest_run() -> dict | None:
    """Get the most recent run log."""
    runs = list_runs()
    if runs:
        return load_run(runs[0])
    return None


def summarize_run(log_data: dict) -> str:
    """Generate a human-readable summary of a run."""
    summary = log_data.get("summary", {})
    config = log_data.get("config", {})
    
    lines = [
        f"# Run Summary: {log_data['run_name']}",
        f"",
        f"**Status**: {log_data['status']}",
        f"**Started**: {log_data['started_at']}",
        f"**Ended**: {log_data.get('ended_at', 'N/A')}",
        f"",
        f"## Stats",
        f"- Total LLM calls: {summary.get('total_llm_calls', 0)}",
        f"- Total tokens: {summary.get('total_tokens', 0):,}",
        f"- Total LLM time: {summary.get('total_llm_duration_sec', 0):.1f}s",
        f"- Generations: {summary.get('total_generations', 0)}",
        f"- Final score: {summary.get('final_score', 'N/A')}",
    ]
    
    # Add evolution history
    history = log_data.get("evolution_history", [])
    if history:
        lines.append("")
        lines.append("## Evolution Progress")
        for gen in history:
            lines.append(f"- Gen {gen['generation']}: best={gen['best_score']:.2f}, avg={gen['avg_score']:.2f}")
    
    # Add any errors
    errors = [e for e in log_data.get("events", []) if e["level"] == "ERROR"]
    if errors:
        lines.append("")
        lines.append("## Errors")
        for err in errors:
            lines.append(f"- {err['timestamp']}: {err['message']}")
    
    return "\n".join(lines)
