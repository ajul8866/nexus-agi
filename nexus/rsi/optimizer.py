"""
NEXUS-AGI Prompt Optimizer
Evolutionary prompt optimization untuk RSI
"""
from __future__ import annotations
import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PromptTemplate:
    id: str
    content: str
    task_type: str
    score: float = 0.5
    trials: int = 0
    created_at: float = field(default_factory=time.time)
    parent_id: Optional[str] = None
    strategy: str = "original"


class PromptOptimizer:
    """Evolutionary prompt optimization dengan A/B testing."""

    STRATEGIES = ["more_specific", "add_examples", "restructure", "add_constraints", "simplify"]

    def __init__(self):
        self._library: Dict[str, List[PromptTemplate]] = {}
        self._pattern_library: Dict[str, List[str]] = {}
        self._generation = 0

    def register_prompt(self, content: str, task_type: str, score: float = 0.5) -> PromptTemplate:
        pt = PromptTemplate(
            id=f"pt_{task_type}_{int(time.time()*1000)}",
            content=content, task_type=task_type, score=score
        )
        self._library.setdefault(task_type, []).append(pt)
        return pt

    def optimize_prompt(self, base_prompt: str, task_type: str,
                        history: Optional[List[Dict]] = None) -> str:
        history = history or []
        templates = self._library.get(task_type, [])
        if templates:
            best = max(templates, key=lambda t: t.score)
            if best.score > 0.7:
                return self._blend_prompts(base_prompt, best.content)
        if history:
            successful = [h for h in history if h.get("success")]
            if successful:
                patterns = self.extract_patterns([h["prompt"] for h in successful])
                if patterns:
                    return self._apply_patterns(base_prompt, patterns)
        strategy = random.choice(self.STRATEGIES)
        return self.generate_variant(base_prompt, strategy)

    def generate_variant(self, prompt: str, strategy: str) -> str:
        if strategy == "more_specific":
            return f"{prompt}\n\nBe specific and precise. Include exact values, names, and formats."
        elif strategy == "add_examples":
            return f"{prompt}\n\nExample output format:\n- Item 1: [value]\n- Item 2: [value]"
        elif strategy == "restructure":
            lines = prompt.strip().split("\n")
            mid = len(lines) // 2
            return "\n".join(lines[mid:] + ["---"] + lines[:mid])
        elif strategy == "add_constraints":
            return f"{prompt}\n\nConstraints:\n- Be concise (max 200 words)\n- Use bullet points\n- No jargon"
        elif strategy == "simplify":
            sentences = prompt.split(".")
            return ". ".join(s.strip() for s in sentences[:3] if s.strip()) + "."
        return prompt

    def ab_test_prompts(self, prompt_a: str, prompt_b: str,
                        scores_a: List[float], scores_b: List[float]) -> Dict[str, Any]:
        avg_a = sum(scores_a) / len(scores_a) if scores_a else 0
        avg_b = sum(scores_b) / len(scores_b) if scores_b else 0
        winner = "A" if avg_a >= avg_b else "B"
        winning_prompt = prompt_a if winner == "A" else prompt_b
        return {"winner": winner, "winning_prompt": winning_prompt,
                "score_a": round(avg_a, 3), "score_b": round(avg_b, 3),
                "improvement": round(abs(avg_a - avg_b), 3)}

    def extract_patterns(self, successful_prompts: List[str]) -> List[str]:
        if not successful_prompts:
            return []
        word_freq: Dict[str, int] = {}
        for prompt in successful_prompts:
            words = prompt.lower().split()
            for w in words:
                if len(w) > 4:
                    word_freq[w] = word_freq.get(w, 0) + 1
        threshold = len(successful_prompts) * 0.5
        patterns = [w for w, count in word_freq.items() if count >= threshold]
        self._pattern_library["common"] = patterns
        return patterns[:10]

    def _blend_prompts(self, base: str, template: str) -> str:
        base_lines = base.strip().split("\n")
        template_lines = template.strip().split("\n")
        blended = base_lines + [""] + template_lines[-max(1, len(template_lines)//4):]
        return "\n".join(blended)

    def _apply_patterns(self, prompt: str, patterns: List[str]) -> str:
        additions = [f"Focus on: {', '.join(patterns[:5])}"] if patterns else []
        return prompt + "\n\n" + "\n".join(additions) if additions else prompt

    def get_best_prompt(self, task_type: str) -> Optional[str]:
        templates = self._library.get(task_type, [])
        if not templates:
            return None
        return max(templates, key=lambda t: t.score).content

    def stats(self) -> Dict[str, Any]:
        return {
            "task_types": list(self._library.keys()),
            "total_templates": sum(len(v) for v in self._library.values()),
            "patterns": len(self._pattern_library),
            "generation": self._generation
        }
