from typing import List, Any
from pydantic import BaseModel, ConfigDict


class StepIdea(BaseModel):
    """
    This model defines a step idea in the reasoning process.
    """
    idea: str
    node_id: int

    model_config = ConfigDict(
        extra='forbid'
    )

class InitialSteps(BaseModel):
    """
    This model defines the initial steps in the reasoning process.
    """
    steps: List[StepIdea]

    model_config = ConfigDict(
        extra='forbid'
    )

class DepthEstimate(BaseModel):
    """
    This model defines the depth estimate in the reasoning process.
    """
    depth: int

    model_config = ConfigDict(
        extra='forbid'
    )

class SubSteps(BaseModel):
    """
    This model defines the sub-steps in the reasoning process.
    """
    sub_steps: List[StepIdea]

    model_config = ConfigDict(
        extra='forbid'
    )

class AnswerResponse(BaseModel):
    """
    This model defines the answer response.
    """
    Answer: str
    Reason: str

    model_config = ConfigDict(
        extra='forbid'
    )

class PrometheusResponse(BaseModel):
    """
    This model defines the Prometheus response.
    """
    feedback: str
    explanation: str

    model_config = ConfigDict(
        extra='forbid'
    )

class ProbingQuestion(BaseModel):
    """
    This model defines the Prometheus probing question.
    """
    probing_question: List[str]

    model_config = ConfigDict(
        extra='forbid'
    )

class LlamaResponse(BaseModel):
    """
    This model defines the Prometheus probing question.
    """
    answer: str

    model_config = ConfigDict(
        extra='forbid'
    )

