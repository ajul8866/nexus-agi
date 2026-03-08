"""ChainOfThought - step-by-step reasoning with backtracking and confidence scoring."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("nexus.planning.chain_of_thought")


@dataclass
class ReasoningStep:
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_number: int = 0
    thought: str = ""
    intermediate_conclusion: str = ""
    confidence: float = 1.0       # 0.0 – 1.0
    evidence: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    is_backtrack: bool = False
    parent_step_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_number,
            "thought": self.thought,
            "conclusion": self.intermediate_conclusion,
            "confidence": round(self.confidence, 3),
            "evidence_count": len(self.evidence),
            "is_backtrack": self.is_backtrack,
        }


@dataclass
class ReasoningChain:
    chain_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    problem: str = ""
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    overall_confidence: float = 0.0
    success: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)

    def chain_confidence(self) -> float:
        """Geometric mean of step confidences (penalises weak links)."""
        if not self.steps:
            return 0.0
        import math
        log_sum = sum(math.log(max(s.confidence, 1e-9)) for s in self.steps)
        return math.exp(log_sum / len(self.steps))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "problem": self.problem,
            "step_count": len(self.steps),
            "final_answer": self.final_answer,
            "overall_confidence": round(self.overall_confidence, 3),
            "chain_confidence": round(self.chain_confidence(), 3),
            "success": self.success,
            "steps": [s.to_dict() for s in self.steps],
        }


class ChainOfThought:
    """
    Step-by-step reasoning engine.
    - Builds reasoning chains with intermediate conclusions
    - Supports backtracking when confidence drops below threshold
    - Tracks alternative hypotheses
    - Scores each step and the overall chain
    """

    def __init__(
        self,
        min_confidence: float = 0.4,
        max_steps: int = 20,
        backtrack_threshold: float = 0.3,
    ):
        self.min_confidence = min_confidence
        self.max_steps = max_steps
        self.backtrack_threshold = backtrack_threshold
        self._chains: Dict[str, ReasoningChain] = {}

    # ── Chain construction ───────────────────────────────────────────────────────────
    def start_chain(self, problem: str) -> ReasoningChain:
        chain = ReasoningChain(problem=problem)
        self._chains[chain.chain_id] = chain
        logger.info("Started CoT chain id=%s problem=%.50s", chain.chain_id, problem)
        return chain

    def add_step(
        self,
        chain: ReasoningChain,
        thought: str,
        conclusion: str,
        confidence: float = 0.8,
        evidence: Optional[List[str]] = None,
        alternatives: Optional[List[str]] = None,
    ) -> ReasoningStep:
        step = ReasoningStep(
            step_number=len(chain.steps) + 1,
            thought=thought,
            intermediate_conclusion=conclusion,
            confidence=max(0.0, min(1.0, confidence)),
            evidence=evidence or [],
            alternatives=alternatives or [],
            parent_step_id=chain.steps[-1].step_id if chain.steps else None,
        )
        chain.steps.append(step)
        logger.debug("CoT step %d confidence=%.2f chain=%s", step.step_number, step.confidence, chain.chain_id)
        return step

    def backtrack(self, chain: ReasoningChain, steps_back: int = 1) -> Optional[ReasoningStep]:
        """Remove the last N steps (backtrack) and mark the new last step."""
        if len(chain.steps) < steps_back:
            logger.warning("Cannot backtrack %d steps; only %d steps exist", steps_back, len(chain.steps))
            return None
        removed = chain.steps[-steps_back:]
        chain.steps = chain.steps[:-steps_back]
        for r in removed:
            r.is_backtrack = True
        if chain.steps:
            chain.steps[-1].alternatives.extend(
                [r.intermediate_conclusion for r in removed]
            )
        logger.info("Backtracked %d steps in chain %s", steps_back, chain.chain_id)
        return chain.steps[-1] if chain.steps else None

    def finalize(self, chain: ReasoningChain, final_answer: str) -> ReasoningChain:
        chain.final_answer = final_answer
        chain.overall_confidence = chain.chain_confidence()
        chain.success = chain.overall_confidence >= self.min_confidence
        return chain

    # ── Automated reasoning ─────────────────────────────────────────────────────────
    def reason(
        self,
        problem: str,
        step_generator: Callable[[str, List[ReasoningStep]], Tuple[str, str, float]],
        max_steps: Optional[int] = None,
    ) -> ReasoningChain:
        """
        Drive reasoning using a provided step_generator callable.
        step_generator(problem, current_steps) -> (thought, conclusion, confidence)
        """
        chain = self.start_chain(problem)
        limit = max_steps or self.max_steps
        backtracks = 0
        max_backtracks = 3

        for _ in range(limit):
            try:
                thought, conclusion, confidence = step_generator(problem, chain.steps)
            except StopIteration:
                break
            except Exception as exc:
                logger.error("Step generator failed: %s", exc)
                break

            step = self.add_step(chain, thought, conclusion, confidence)

            if confidence < self.backtrack_threshold and backtracks < max_backtracks:
                self.backtrack(chain, steps_back=1)
                backtracks += 1
                logger.info("Low confidence %.2f; backtracking (attempt %d)", confidence, backtracks)
                continue

            if conclusion.lower().strip().startswith("final answer:") or confidence > 0.95:
                break

        final = chain.steps[-1].intermediate_conclusion if chain.steps else "No conclusion reached"
        return self.finalize(chain, final)

    # ── Simple built-in reasoner (decomposition-based) ─────────────────────────────
    def decompose_and_reason(self, problem: str) -> ReasoningChain:
        """Built-in heuristic reasoner for demo/testing."""
        chain = self.start_chain(problem)
        words = problem.split()

        steps_data = [
            (
                f"Understand the problem: '{problem[:60]}'",
                f"The problem involves {len(words)} words and asks about: {' '.join(words[:5])}",
                0.9,
            ),
            (
                "Identify key components and constraints",
                f"Key entities: {', '.join(set(w for w in words if len(w) > 4)[:5]) or 'none identified'}",
                0.75,
            ),
            (
                "Consider relevant knowledge and approaches",
                "Applying systematic decomposition and elimination strategies",
                0.8,
            ),
            (
                "Synthesise intermediate findings",
                f"Based on analysis, the problem can be addressed through structured reasoning",
                0.85,
            ),
        ]

        for thought, conclusion, confidence in steps_data:
            self.add_step(chain, thought, conclusion, confidence)

        final = f"Final answer: Systematic analysis of '{problem[:40]}' completed with {len(chain.steps)} reasoning steps."
        return self.finalize(chain, final)

    # ── History ──────────────────────────────────────────────────────────────────
    def get_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        return self._chains.get(chain_id)

    def get_all_chains(self) -> List[ReasoningChain]:
        return list(self._chains.values())

    def stats(self) -> Dict[str, Any]:
        chains = list(self._chains.values())
        successful = [c for c in chains if c.success]
        return {
            "total_chains": len(chains),
            "successful": len(successful),
            "success_rate": len(successful) / max(len(chains), 1),
            "avg_steps": sum(len(c.steps) for c in chains) / max(len(chains), 1),
            "avg_confidence": sum(c.overall_confidence for c in chains) / max(len(chains), 1),
        }
