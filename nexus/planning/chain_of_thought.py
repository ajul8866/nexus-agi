"""ChainOfThought - structured step-by-step reasoning engine."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nexus.planning.chain_of_thought")


@dataclass
class ReasoningStep:
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_number: int = 0
    thought: str = ""
    observation: str = ""
    action: Optional[str] = None
    action_input: Optional[Any] = None
    action_output: Optional[Any] = None
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_final: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"step": self.step_number, "thought": self.thought, "observation": self.observation, "action": self.action, "confidence": self.confidence, "is_final": self.is_final}


@dataclass
class ReasoningChain:
    chain_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed: bool = False
    total_tokens_est: int = 0

    def add_step(self, step: ReasoningStep) -> None:
        step.step_number = len(self.steps) + 1
        self.steps.append(step)

    def avg_confidence(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.confidence for s in self.steps) / len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {"chain_id": self.chain_id, "question": self.question, "steps": [s.to_dict() for s in self.steps], "final_answer": self.final_answer, "completed": self.completed, "avg_confidence": round(self.avg_confidence(), 3), "step_count": len(self.steps)}

    def format_prompt(self) -> str:
        lines = [f"Question: {self.question}"]
        for step in self.steps:
            lines.append(f"\nThought {step.step_number}: {step.thought}")
            if step.action:
                lines.append(f"Action: {step.action}")
                lines.append(f"Action Input: {step.action_input}")
                lines.append(f"Observation: {step.observation}")
        return "\n".join(lines)


class ChainOfThought:
    """
    Chain-of-Thought reasoning engine.
    Manages multi-step reasoning chains with ReAct-style (Reason + Act) loops,
    confidence tracking, backtracking, and answer synthesis.
    """

    MAX_STEPS = 20

    def __init__(self, llm_callable=None, tools: Optional[Dict[str, Any]] = None, max_steps: int = 10, min_confidence: float = 0.3):
        self.llm = llm_callable
        self.tools = tools or {}
        self.max_steps = min(max_steps, self.MAX_STEPS)
        self.min_confidence = min_confidence
        self._chains: Dict[str, ReasoningChain] = {}

    def start_chain(self, question: str) -> ReasoningChain:
        chain = ReasoningChain(question=question)
        self._chains[chain.chain_id] = chain
        logger.info("Started reasoning chain %s: %s", chain.chain_id, question[:60])
        return chain

    def add_thought(self, chain: ReasoningChain, thought: str, confidence: float = 1.0, action: Optional[str] = None, action_input: Optional[Any] = None) -> ReasoningStep:
        step = ReasoningStep(thought=thought, confidence=confidence, action=action, action_input=action_input)
        chain.add_step(step)
        return step

    def execute_action(self, step: ReasoningStep) -> Any:
        if not step.action or step.action not in self.tools:
            step.observation = f"Tool '{step.action}' not available."
            return None
        try:
            tool_fn = self.tools[step.action]
            result = tool_fn(step.action_input)
            step.action_output = result
            step.observation = str(result)[:500]
            return result
        except Exception as exc:
            step.observation = f"Error: {exc}"
            step.confidence *= 0.5
            return None

    def finalize(self, chain: ReasoningChain, answer: str, confidence: float = 1.0) -> ReasoningStep:
        final_step = ReasoningStep(thought=f"Final answer: {answer}", confidence=confidence, is_final=True)
        chain.add_step(final_step)
        chain.final_answer = answer
        chain.completed = True
        logger.info("Chain %s completed: %d steps, avg_conf=%.3f", chain.chain_id, len(chain.steps), chain.avg_confidence())
        return final_step

    def backtrack(self, chain: ReasoningChain, steps: int = 1) -> int:
        removed = min(steps, len(chain.steps))
        chain.steps = chain.steps[:-removed]
        logger.debug("Backtracked %d steps in chain %s", removed, chain.chain_id)
        return removed

    def reason(self, question: str, context: Optional[str] = None) -> ReasoningChain:
        chain = self.start_chain(question)
        # Simple rule-based CoT (placeholder for LLM integration)
        thoughts = self._generate_thoughts(question, context)
        for i, (thought, action, inp) in enumerate(thoughts):
            step = self.add_thought(chain, thought, confidence=0.8, action=action, action_input=inp)
            if action:
                self.execute_action(step)
            if i >= self.max_steps - 1:
                break
        answer = self._synthesize_answer(chain)
        self.finalize(chain, answer, confidence=chain.avg_confidence())
        return chain

    def _generate_thoughts(self, question: str, context: Optional[str]) -> List[tuple]:
        q = question.lower()
        thoughts = [(f"Let me analyse the question: '{question}'", None, None)]
        if context:
            thoughts.append((f"Given the context: {context[:100]}", None, None))
        if "calculate" in q or "compute" in q or "how many" in q:
            thoughts.append(("This requires a calculation.", "calculate", question))
        elif "search" in q or "find" in q or "what is" in q:
            thoughts.append(("I need to search for information.", "search", question))
        elif "compare" in q:
            thoughts.append(("I need to compare options.", None, None))
            thoughts.append(("Breaking down each option for comparison.", None, None))
        else:
            thoughts.append(("Let me reason step by step.", None, None))
            thoughts.append(("Considering all relevant factors.", None, None))
        thoughts.append(("I have enough information to formulate an answer.", None, None))
        return thoughts

    def _synthesize_answer(self, chain: ReasoningChain) -> str:
        observations = [s.observation for s in chain.steps if s.observation]
        if observations:
            return f"Based on my reasoning ({len(chain.steps)} steps): {observations[-1]}"
        return f"After {len(chain.steps)} reasoning steps, I have analysed the question: '{chain.question}'"

    def get_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        return self._chains.get(chain_id)

    def stats(self) -> Dict[str, Any]:
        completed = [c for c in self._chains.values() if c.completed]
        return {"total_chains": len(self._chains), "completed": len(completed), "avg_steps": round(sum(len(c.steps) for c in completed) / max(len(completed), 1), 2), "avg_confidence": round(sum(c.avg_confidence() for c in completed) / max(len(completed), 1), 3)}
