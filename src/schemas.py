from enum import Enum

from pydantic import BaseModel, Field


class Domain(str, Enum):
    INFANT_CARE = "Infant Care"
    PHYSICAL_RECOVERY = "Postpartum Physical Recovery"
    MENTAL_HEALTH = "Postpartum Mental Health Recovery"


class LiteracyLevel(str, Enum):
    VERY_LOW = "Very Low"
    INADEQUATE = "Inadequate"
    MARGINAL = "Marginal"
    ADEQUATE = "Adequate"


class QuestionInput(BaseModel):
    question_id: int
    text: str
    domain: Domain
    literacy_level: LiteracyLevel


class RevisionProposal(BaseModel):
    agent_name: str

    revised_question: str = Field(
        description="The agent's proposed revised patient question."
    )

    meaning_preserved: bool = Field(
        description="Whether the original clinical meaning was preserved."
    )

    unsupported_information_added: bool = Field(
        description=(
            "Whether the revision introduced information "
            "not stated in the original patient question."
        )
    )

    brief_note: str = Field(
        description="Short explanation of the revision."
    )


class Vote(BaseModel):
    agent_name: str

    selected_candidate_id: str = Field(
        description="Anonymous candidate ID selected, such as A, B, or C."
    )

    brief_reason: str = Field(
        description="Short reason for selecting the candidate."
    )


class FinalSelection(BaseModel):
    revised_question: str

    decision_method: str = Field(
        description=(
            "How the final question was selected, such as "
            "consensus or blind_vote."
        )
    )

    winning_candidate_id: str | None = None

    source_agents: list[str] = Field(
        default_factory=list
    )

    vote_counts: dict[str, int] = Field(
        default_factory=dict
    )