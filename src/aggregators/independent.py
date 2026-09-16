from src.llm.base import LLMClient
from src.schemas import (
    QuestionInput,
    RevisionProposal,
)


class IndependentAggregator:
    def __init__(
        self,
        llm: LLMClient,
    ):
        self.llm = llm

    def synthesize(
        self,
        question: QuestionInput,
        proposals: list[RevisionProposal],
    ) -> RevisionProposal:

        proposal_text = "\n\n".join(
            [
                f"""
INDEPENDENT REVISION {index}:
{proposal.revised_question}
"""
                for index, proposal in enumerate(
                    proposals,
                    start=1,
                )
            ]
        )

        prompt = f"""
You are the synthesis component of an independent
multi-agent healthcare communication system.

The clinical agents generated their revisions independently.
They did not communicate with one another.

ORIGINAL PATIENT QUESTION:
{question.text}

MEDICAL DOMAIN:
{question.domain.value}

HEALTH LITERACY LEVEL:
{question.literacy_level.value}

INDEPENDENT REVISIONS:
{proposal_text}

TASK:
Produce one final patient-facing question from the independent
revisions.

IMPORTANT:
You are performing synthesis only.

Do NOT:
- rank the agents,
- vote between revisions,
- critique an individual agent,
- introduce new clinical information,
- choose a medically more detailed interpretation.

The ORIGINAL PATIENT QUESTION is authoritative.

Your output must:
1. Preserve the patient's original clinical intent.
2. Preserve the original question type.
3. Use only information supported by the original question.
4. Avoid adding symptoms, diagnoses, timing, severity,
   causes, medications, or medical history.
5. Improve clarity while remaining patient-like.

Return:
- one synthesized patient question,
- whether the original meaning was preserved,
- whether unsupported information was added,
- and a short note describing the synthesis.
"""

        result = self.llm.generate_structured(
            prompt=prompt,
            output_schema=RevisionProposal,
        )

        if result is None:
            raise ValueError(
                "Independent aggregator did not return "
                "a valid RevisionProposal."
            )

        result.agent_name = "Independent Aggregator"

        return result