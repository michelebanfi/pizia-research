"""
AlphaEvolve-style Evolution Loop.

Implements the core evolutionary algorithm:
1. Generate: Spawn parallel code proposals with cheap models
2. Evaluate: Run proposals in sandbox against tests
3. Select: Keep best performing code
4. Iterate: Feed feedback back to generator
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncGenerator, Callable

from config import (
    EVOLUTION_PARALLEL_CANDIDATES,
    EVOLUTION_MAX_GENERATIONS,
    EVOLUTION_EARLY_STOP_SCORE,
    get_default_cheap_model,
)
from utils.llm_engine import LLMEngine, LLMResponse, get_engine
from utils.sandbox import ExecutionResult, run
from utils.logger import get_logger


@dataclass
class Solution:
    """A candidate solution in the population."""
    code: str
    score: float
    tests_passed: int
    tests_total: int
    feedback: str
    generation: int
    execution_time: float = 0.0
    parent_code: str | None = None  # For tracking lineage


@dataclass
class EvolutionState:
    """Current state of the evolution process."""
    generation: int
    best_solution: Solution | None
    population: list[Solution]
    history: list[Solution]  # All solutions ever tried
    status: str  # "running", "success", "max_generations", "error"
    start_time: datetime = field(default_factory=datetime.now)


@dataclass
class EvolutionResult:
    """Final result of evolution."""
    best_solution: Solution | None
    total_generations: int
    total_candidates_evaluated: int
    success: bool
    history: list[Solution]
    duration_seconds: float


# =============================================================================
# Prompts
# =============================================================================

INITIAL_CODE_PROMPT = """You are an expert Python programmer. Generate a solution for the following problem.

## Problem Context
{context}

## Requirements
{requirements}

## Task
Write a complete Python solution. The code should be self-contained and implement all necessary functions.

Return ONLY the Python code, no explanations or markdown formatting."""


EVOLUTION_PROMPT = """You are an evolutionary coding agent. Your task is to improve the given code.

## Problem Context
{context}

## Current Code
```python
{current_code}
```

## Test Results
Score: {score:.1%} ({tests_passed}/{tests_total} tests passed)

## Feedback/Errors
{feedback}

## Task
Improve the code to pass more tests. Focus on fixing the errors shown above.
If the code has syntax errors, fix them.
If tests are failing, analyze why and fix the logic.

**IMPORTANT**: When something doesn't work, think about WHY it doesn't work.
Understanding the root cause helps expand our knowledge of the problem.
If an approach fails, briefly note the reason before proposing the fix.

Return ONLY the improved Python code, no explanations."""


DIFF_EVOLUTION_PROMPT = """You are an evolutionary coding agent. Suggest improvements using SEARCH/REPLACE blocks.

## Problem Context
{context}

## Current Code
```python
{current_code}
```

## Test Results
Score: {score:.1%} ({tests_passed}/{tests_total} tests passed)

## Errors
{feedback}

## Task
Suggest specific changes to fix the errors. Use this format:

<<<<<<< SEARCH
exact code to find
=======
replacement code
>>>>>>> REPLACE

**IMPORTANT**: When something doesn't work, explain WHY it doesn't work.
Understanding the root cause deepens our knowledge of the problem domain.
Before each fix, briefly note the underlying reason for the failure.

You can include multiple SEARCH/REPLACE blocks for different fixes.
Be precise with the search text - it must match exactly."""


# =============================================================================
# Population Management
# =============================================================================

class Population:
    """
    Manages the population of candidate solutions.
    
    Implements selection strategies for the evolutionary algorithm.
    """
    
    def __init__(self, max_size: int = 10):
        self.solutions: list[Solution] = []
        self.max_size = max_size
        self._best: Solution | None = None
    
    @property
    def best(self) -> Solution | None:
        """Get the best solution seen so far."""
        return self._best
    
    def add(self, solution: Solution) -> None:
        """Add a solution to the population."""
        self.solutions.append(solution)
        
        # Update best
        if self._best is None or solution.score > self._best.score:
            self._best = solution
        
        # Keep population bounded
        if len(self.solutions) > self.max_size:
            # Remove worst solutions
            self.solutions.sort(key=lambda s: s.score, reverse=True)
            self.solutions = self.solutions[:self.max_size]
    
    def get_parent(self) -> Solution | None:
        """
        Select a parent solution for the next generation.
        
        Uses a mix of exploitation (best) and exploration (random from top half).
        """
        if not self.solutions:
            return None
        
        # 70% chance to use best, 30% chance to explore
        import random
        if random.random() < 0.7 or len(self.solutions) == 1:
            return self._best
        
        # Select from top half
        sorted_solutions = sorted(self.solutions, key=lambda s: s.score, reverse=True)
        top_half = sorted_solutions[:max(1, len(sorted_solutions) // 2)]
        return random.choice(top_half)
    
    def get_diverse_parents(self, n: int) -> list[Solution]:
        """Get n diverse parent solutions for parallel generation."""
        if not self.solutions:
            return []
        
        if len(self.solutions) <= n:
            return self.solutions.copy()
        
        # Mix of best and random
        result = [self._best] if self._best else []
        remaining = [s for s in self.solutions if s != self._best]
        
        import random
        random.shuffle(remaining)
        result.extend(remaining[:n - len(result)])
        
        return result


# =============================================================================
# Diff Application
# =============================================================================

def apply_diff(code: str, diff_content: str) -> str:
    """
    Apply SEARCH/REPLACE diff blocks to code.
    
    Format:
    <<<<<<< SEARCH
    text to find
    =======
    replacement text
    >>>>>>> REPLACE
    """
    result = code
    
    # Find all diff blocks
    pattern = r'<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE'
    matches = re.findall(pattern, diff_content, re.DOTALL)
    
    for search, replace in matches:
        if search.strip() in result:
            result = result.replace(search.strip(), replace.strip(), 1)
    
    return result


def extract_code(response: str) -> str:
    """Extract code from LLM response, handling markdown blocks."""
    code = response
    
    # Handle markdown code blocks
    if "```python" in code:
        parts = code.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
    elif "```" in code:
        parts = code.split("```")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
    
    return code.strip()


# =============================================================================
# Evolver Agent
# =============================================================================

class EvolverAgent:
    """
    The AlphaEvolve-style evolution loop.
    
    Iteratively improves code by:
    1. Generating parallel proposals with cheap models
    2. Evaluating in sandbox
    3. Selecting best performers
    4. Feeding feedback back to generator
    """
    
    def __init__(
        self,
        engine: LLMEngine | None = None,
        parallel_candidates: int = EVOLUTION_PARALLEL_CANDIDATES,
        max_generations: int = EVOLUTION_MAX_GENERATIONS,
        early_stop_score: float = EVOLUTION_EARLY_STOP_SCORE,
    ):
        self.engine = engine or get_engine()
        self.parallel_candidates = parallel_candidates
        self.max_generations = max_generations
        self.early_stop_score = early_stop_score
        self.model = get_default_cheap_model()
    
    async def evolve(
        self,
        context: str,
        tests: str,
        initial_code: str | None = None,
        on_generation: Callable[[EvolutionState], None] | None = None,
    ) -> EvolutionResult:
        """
        Run the full evolution loop.
        
        Args:
            context: The problem context (from researcher)
            tests: The test code (from researcher)
            initial_code: Optional starting code
            on_generation: Callback for each generation
            
        Returns:
            EvolutionResult with best solution and history
        """
        population = Population()
        history: list[Solution] = []
        start_time = datetime.now()
        
        # Generate initial solution if not provided
        if not initial_code:
            initial_code = await self._generate_initial(context)
        
        # Evaluate initial solution
        initial_result = await run(initial_code, tests)
        initial_solution = Solution(
            code=initial_code,
            score=initial_result.score,
            tests_passed=initial_result.tests_passed,
            tests_total=initial_result.tests_total,
            feedback=initial_result.errors or initial_result.output,
            generation=0,
            execution_time=initial_result.execution_time,
        )
        population.add(initial_solution)
        history.append(initial_solution)
        
        state = EvolutionState(
            generation=0,
            best_solution=initial_solution,
            population=population.solutions.copy(),
            history=history,
            status="running",
        )
        
        if on_generation:
            on_generation(state)
        
        # Check early stop
        if initial_solution.score >= self.early_stop_score:
            return self._create_result(population, history, start_time, "success")
        
        # Evolution loop
        for gen in range(1, self.max_generations + 1):
            # Generate candidates in parallel
            parent = population.get_parent()
            if not parent:
                break
            
            candidates = await self._generate_candidates(
                context=context,
                parent=parent,
                n=self.parallel_candidates,
            )
            
            # Evaluate all candidates
            best_this_gen: Solution | None = None
            
            for candidate_code in candidates:
                if not candidate_code.strip():
                    continue
                
                result = await run(candidate_code, tests)
                
                solution = Solution(
                    code=candidate_code,
                    score=result.score,
                    tests_passed=result.tests_passed,
                    tests_total=result.tests_total,
                    feedback=result.errors or result.output,
                    generation=gen,
                    execution_time=result.execution_time,
                    parent_code=parent.code,
                )
                
                population.add(solution)
                history.append(solution)
                
                if best_this_gen is None or solution.score > best_this_gen.score:
                    best_this_gen = solution
            
            # Update state
            state = EvolutionState(
                generation=gen,
                best_solution=population.best,
                population=population.solutions.copy(),
                history=history,
                status="running",
            )
            
            if on_generation:
                on_generation(state)
            
            # Log generation progress
            logger = get_logger()
            if logger and best_this_gen:
                all_scores = [s.score for s in history if s.generation == gen]
                logger.log_generation(
                    generation=gen,
                    candidates_count=len(candidates),
                    best_score=population.best.score if population.best else 0,
                    best_code_preview=population.best.code if population.best else "",
                    all_scores=all_scores,
                )
            
            # Check early stop
            if population.best and population.best.score >= self.early_stop_score:
                return self._create_result(population, history, start_time, "success")
        
        # Max generations reached
        return self._create_result(population, history, start_time, "max_generations")
    
    async def evolve_stream(
        self,
        context: str,
        tests: str,
        initial_code: str | None = None,
    ) -> AsyncGenerator[EvolutionState, None]:
        """
        Stream evolution progress as an async generator.
        
        Yields EvolutionState after each generation.
        """
        states: list[EvolutionState] = []
        
        def capture_state(state: EvolutionState):
            states.append(state)
        
        # Run evolution with callback (this is a bit hacky but works)
        task = asyncio.create_task(
            self.evolve(context, tests, initial_code, capture_state)
        )
        
        last_yielded = 0
        while not task.done():
            await asyncio.sleep(0.1)
            while last_yielded < len(states):
                yield states[last_yielded]
                last_yielded += 1
        
        # Yield any remaining states
        while last_yielded < len(states):
            yield states[last_yielded]
            last_yielded += 1
    
    async def _generate_initial(self, context: str) -> str:
        """Generate initial code solution."""
        prompt = INITIAL_CODE_PROMPT.format(
            context=context[:6000],
            requirements="Write clean, working Python code.",
        )
        
        response = await self.engine.generate_with_cheap_model(prompt)
        
        if response.success:
            return extract_code(response.content)
        
        return "# Initial code generation failed\npass"
    
    async def _generate_candidates(
        self,
        context: str,
        parent: Solution,
        n: int,
    ) -> list[str]:
        """Generate n candidate solutions in parallel."""
        # Build prompts with slight variations for diversity
        prompts = []
        for i in range(n):
            # Vary the prompt slightly for diversity
            temperature_hint = ""
            if i == 0:
                temperature_hint = "Focus on fixing the exact error."
            elif i == 1:
                temperature_hint = "Try a different algorithmic approach."
            elif i == 2:
                temperature_hint = "Simplify the code structure."
            else:
                temperature_hint = f"Variation {i}: try something creative."
            
            prompt = EVOLUTION_PROMPT.format(
                context=context[:4000],
                current_code=parent.code,
                score=parent.score,
                tests_passed=parent.tests_passed,
                tests_total=parent.tests_total,
                feedback=parent.feedback[:2000] if parent.feedback else "None",
            ) + f"\n\nHint: {temperature_hint}"
            
            prompts.append(prompt)
        
        # Generate in parallel
        responses = await self.engine.parallel_generate(
            model=self.model,
            prompts=prompts,
            system_prompt="You are an expert Python programmer. Return only code.",
            temperature=0.8,  # Higher temp for diversity
        )
        
        # Extract code from responses
        candidates = []
        for resp in responses:
            if resp.success:
                candidates.append(extract_code(resp.content))
        
        return candidates
    
    def _create_result(
        self,
        population: Population,
        history: list[Solution],
        start_time: datetime,
        status: str,
    ) -> EvolutionResult:
        """Create final evolution result."""
        duration = (datetime.now() - start_time).total_seconds()
        
        return EvolutionResult(
            best_solution=population.best,
            total_generations=max(s.generation for s in history) if history else 0,
            total_candidates_evaluated=len(history),
            success=status == "success",
            history=history,
            duration_seconds=duration,
        )


# =============================================================================
# Convenience function
# =============================================================================

async def run_evolution(
    context: str,
    tests: str,
    max_generations: int = EVOLUTION_MAX_GENERATIONS,
    parallel_candidates: int = EVOLUTION_PARALLEL_CANDIDATES,
) -> EvolutionResult:
    """
    Convenience function to run evolution.
    
    Args:
        context: Problem context
        tests: Test code
        max_generations: Maximum generations
        parallel_candidates: Candidates per generation
        
    Returns:
        EvolutionResult with best solution
    """
    agent = EvolverAgent(
        max_generations=max_generations,
        parallel_candidates=parallel_candidates,
    )
    return await agent.evolve(context, tests)


# =============================================================================
# Example usage
# =============================================================================

async def _demo():
    """Demo the evolver functionality."""
    
    context = """
    Write a Python function called `reverse_string` that takes a string
    and returns it reversed.
    
    Example:
    - reverse_string("hello") -> "olleh"
    - reverse_string("") -> ""
    - reverse_string("a") -> "a"
    """
    
    tests = '''
def run_tests():
    """Test the reverse_string function."""
    passed = 0
    total = 4
    details = []
    
    # Test 1: Normal string
    try:
        result = reverse_string("hello")
        if result == "olleh":
            passed += 1
            details.append("Test 1: PASS - reverse_string('hello') = 'olleh'")
        else:
            details.append(f"Test 1: FAIL - got {result!r}, expected 'olleh'")
    except Exception as e:
        details.append(f"Test 1: ERROR - {e}")
    
    # Test 2: Empty string
    try:
        result = reverse_string("")
        if result == "":
            passed += 1
            details.append("Test 2: PASS - reverse_string('') = ''")
        else:
            details.append(f"Test 2: FAIL - got {result!r}, expected ''")
    except Exception as e:
        details.append(f"Test 2: ERROR - {e}")
    
    # Test 3: Single char
    try:
        result = reverse_string("a")
        if result == "a":
            passed += 1
            details.append("Test 3: PASS - reverse_string('a') = 'a'")
        else:
            details.append(f"Test 3: FAIL - got {result!r}, expected 'a'")
    except Exception as e:
        details.append(f"Test 3: ERROR - {e}")
    
    # Test 4: Palindrome
    try:
        result = reverse_string("racecar")
        if result == "racecar":
            passed += 1
            details.append("Test 4: PASS - reverse_string('racecar') = 'racecar'")
        else:
            details.append(f"Test 4: FAIL - got {result!r}, expected 'racecar'")
    except Exception as e:
        details.append(f"Test 4: ERROR - {e}")
    
    return {"passed": passed, "total": total, "details": "\\n".join(details)}
'''
    
    print("Running evolution demo...")
    print("=" * 50)
    
    def on_gen(state: EvolutionState):
        print(f"Generation {state.generation}: Best score = {state.best_solution.score:.1%}" if state.best_solution else "No solution yet")
    
    agent = EvolverAgent(max_generations=5, parallel_candidates=3)
    result = await agent.evolve(context, tests, on_generation=on_gen)
    
    print("\n" + "=" * 50)
    print(f"Evolution complete!")
    print(f"Success: {result.success}")
    print(f"Generations: {result.total_generations}")
    print(f"Candidates evaluated: {result.total_candidates_evaluated}")
    print(f"Duration: {result.duration_seconds:.2f}s")
    
    if result.best_solution:
        print(f"\nBest score: {result.best_solution.score:.1%}")
        print(f"Best code:\n{result.best_solution.code}")


if __name__ == "__main__":
    asyncio.run(_demo())
