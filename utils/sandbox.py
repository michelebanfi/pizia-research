"""
Secure Code Execution Sandbox.

Executes Python code in isolated subprocess with timeout and output capture.
Based on patterns from poetiq-arc-agi-solver/arc_agi/sandbox.py.
"""

import asyncio
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from typing import Any

from config import SANDBOX_TIMEOUT_SECONDS, SANDBOX_MAX_OUTPUT_CHARS


@dataclass
class ExecutionResult:
    """Result from code execution."""
    success: bool
    output: str
    errors: str
    score: float  # 0.0 to 1.0, based on test pass rate
    tests_passed: int
    tests_total: int
    execution_time: float
    raw_result: dict[str, Any] | None = None


def _run_subprocess_sync(script_path: str, cwd: str, timeout: float) -> tuple[int, str, str, float]:
    """
    Run the subprocess synchronously. This is called from a thread pool
    to avoid Windows asyncio subprocess issues.
    
    Returns: (returncode, stdout, stderr, execution_time)
    """
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            cwd=cwd,
            timeout=timeout,
            env={
                **os.environ,
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        
        execution_time = time.time() - start_time
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        
        return (result.returncode, stdout, stderr, execution_time)
        
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return (-1, "", "Execution timed out", execution_time)


async def run(
    code: str,
    tests: str,
    timeout: float = SANDBOX_TIMEOUT_SECONDS,
) -> ExecutionResult:
    """
    Run user code against test cases in an isolated subprocess.
    
    Uses synchronous subprocess.run in a thread pool to avoid Windows
    asyncio subprocess compatibility issues.
    
    Args:
        code: The Python code to execute
        tests: The test code (should define run_tests function)
        timeout: Maximum execution time in seconds
        
    Returns:
        ExecutionResult with success status, outputs, and score
    """
    # Build the complete script
    script = _build_script(code, tests)
    
    with tempfile.TemporaryDirectory() as td:
        script_path = os.path.join(td, "solution.py")
        
        # Write the script to temp file
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        
        # Run subprocess in thread pool (avoids Windows asyncio issues)
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            returncode, stdout_text, stderr_text, execution_time = await loop.run_in_executor(
                pool,
                _run_subprocess_sync,
                script_path,
                td,
                timeout,
            )
        
        # Trim output
        stdout_text = stdout_text[:SANDBOX_MAX_OUTPUT_CHARS]
        stderr_text = stderr_text[:SANDBOX_MAX_OUTPUT_CHARS]
        
        # Check for timeout
        if returncode == -1 and "timed out" in stderr_text:
            return ExecutionResult(
                success=False,
                output="",
                errors="Execution timed out",
                score=0.0,
                tests_passed=0,
                tests_total=0,
                execution_time=timeout,
            )
        
        # Check for runtime errors
        if returncode != 0:
            return ExecutionResult(
                success=False,
                output=stdout_text,
                errors=stderr_text or f"Exit code: {returncode}",
                score=0.0,
                tests_passed=0,
                tests_total=0,
                execution_time=execution_time,
            )
        
        # Parse the JSON result from stdout
        try:
            # Find the JSON result line (last line should be our result)
            lines = stdout_text.strip().split("\n")
            result_line = None
            for line in reversed(lines):
                if line.startswith("{"):
                    result_line = line
                    break
            
            if not result_line:
                return ExecutionResult(
                    success=False,
                    output=stdout_text,
                    errors="No JSON result found in output",
                    score=0.0,
                    tests_passed=0,
                    tests_total=0,
                    execution_time=execution_time,
                )
            
            payload = json.loads(result_line)
            
            tests_passed = payload.get("tests_passed", 0)
            tests_total = payload.get("tests_total", 1)
            score = tests_passed / tests_total if tests_total > 0 else 0.0
            
            return ExecutionResult(
                success=payload.get("ok", False),
                output=payload.get("output", stdout_text),
                errors=payload.get("errors", ""),
                score=score,
                tests_passed=tests_passed,
                tests_total=tests_total,
                execution_time=execution_time,
                raw_result=payload,
            )
            
        except json.JSONDecodeError as e:
            return ExecutionResult(
                success=False,
                output=stdout_text,
                errors=f"Failed to parse result JSON: {e}",
                score=0.0,
                tests_passed=0,
                tests_total=0,
                execution_time=execution_time,
            )


def _build_script(code: str, tests: str) -> str:
    """
    Build the complete execution script.
    
    Wraps user code and tests in a harness that:
    1. Catches all exceptions
    2. Runs tests and counts pass/fail
    3. Outputs structured JSON result
    """
    return textwrap.dedent(f'''
# ============================================================================
# SOLUTION CODE
# ============================================================================
{code}

# ============================================================================
# TEST CODE
# ============================================================================
{tests}

# ============================================================================
# TEST HARNESS
# ============================================================================
if __name__ == "__main__":
    import json
    import traceback
    
    results = {{
        "ok": False,
        "tests_passed": 0,
        "tests_total": 0,
        "output": "",
        "errors": "",
    }}
    
    try:
        # Run the tests - expects run_tests() to return a dict with:
        # {{"passed": int, "total": int, "details": str}}
        if "run_tests" in dir():
            test_result = run_tests()
            results["tests_passed"] = test_result.get("passed", 0)
            results["tests_total"] = test_result.get("total", 0)
            results["output"] = test_result.get("details", "")
            results["ok"] = results["tests_passed"] == results["tests_total"]
        else:
            results["errors"] = "No run_tests() function defined"
            
    except Exception as e:
        results["errors"] = traceback.format_exc()
    
    print(json.dumps(results))
''')


async def run_simple(
    code: str,
    timeout: float = SANDBOX_TIMEOUT_SECONDS,
) -> ExecutionResult:
    """
    Run code without tests - just execute and capture output.
    
    Useful for testing code generation without a test harness.
    """
    simple_tests = textwrap.dedent('''
def run_tests():
    """No-op test that just checks code executed."""
    return {"passed": 1, "total": 1, "details": "Code executed successfully"}
''')
    return await run(code, simple_tests, timeout)


# =============================================================================
# Example usage
# =============================================================================

async def _demo():
    """Demo the sandbox functionality."""
    
    # Test with passing code
    code = textwrap.dedent('''
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
''')
    
    tests = textwrap.dedent('''
def run_tests():
    passed = 0
    total = 3
    details = []
    
    # Test 1
    if add(2, 3) == 5:
        passed += 1
        details.append("add(2, 3) = 5 ✓")
    else:
        details.append("add(2, 3) != 5 ✗")
    
    # Test 2
    if add(-1, 1) == 0:
        passed += 1
        details.append("add(-1, 1) = 0 ✓")
    else:
        details.append("add(-1, 1) != 0 ✗")
    
    # Test 3
    if multiply(3, 4) == 12:
        passed += 1
        details.append("multiply(3, 4) = 12 ✓")
    else:
        details.append("multiply(3, 4) != 12 ✗")
    
    return {"passed": passed, "total": total, "details": "\\n".join(details)}
''')
    
    print("Testing passing code...")
    result = await run(code, tests)
    print(f"Success: {result.success}")
    print(f"Score: {result.score:.1%}")
    print(f"Tests: {result.tests_passed}/{result.tests_total}")
    print(f"Time: {result.execution_time:.3f}s")
    print(f"Output:\n{result.output}")
    
    print("\n" + "="*50 + "\n")
    
    # Test with failing code
    bad_code = textwrap.dedent('''
def add(a, b):
    return a - b  # Bug!

def multiply(a, b):
    return a * b
''')
    
    print("Testing failing code...")
    result = await run(bad_code, tests)
    print(f"Success: {result.success}")
    print(f"Score: {result.score:.1%}")
    print(f"Tests: {result.tests_passed}/{result.tests_total}")
    print(f"Output:\n{result.output}")
    
    print("\n" + "="*50 + "\n")
    
    # Test with syntax error
    error_code = "def broken(:\n    return"
    
    print("Testing code with syntax error...")
    result = await run(error_code, tests)
    print(f"Success: {result.success}")
    print(f"Errors:\n{result.errors[:200]}")


if __name__ == "__main__":
    asyncio.run(_demo())
