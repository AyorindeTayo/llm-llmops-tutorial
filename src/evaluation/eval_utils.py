"""Evaluation data structures — importable for tests."""
from dataclasses import dataclass
from typing import List


@dataclass
class EvalSample:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str


@dataclass
class EvalResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def overall_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4
