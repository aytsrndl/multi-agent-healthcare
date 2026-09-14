from src.llm.base import LLMClient
from src.schemas import (
    QuestionInput,
    RevisionProposal,
)


class CentralizedOrchestrator:
    """
    Central orchestrator for the paper-aligned
    Centralized MAS condition.

    Clinical agents independently generate candidate revisions.
    They do not communicate directly with one another.

    The orchestrator receives all worker outputs, evaluates them
    against the original patient question, and produces the final
    revised question.
    """

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

        worker_text = "\n\n".join(
            [
                f"""
WORKER {index}
ROLE: {proposal.agent_name}
PROPOSED QUESTION:
{proposal.revised_question}

WORKER NOTE:
{proposal.brief_note}
"""
                for index, proposal in enumerate(
                    proposals,
                    start=1,
                )
            ]
        )

        prompt = f"""
You are the central orchestrator in a healthcare
communication multi-agent system.

Clinical worker agents independently revised a patient's question.
The workers did not communicate directly with one another.

ORIGINAL PATIENT QUESTION:
{question.text}

MEDICAL DOMAIN:
{question.domain.value}

HEALTH LITERACY LEVEL:
{question.literacy_level.value}

WORKER PROPOSALS:
{worker_text}

TASK:
Evaluate the worker proposals and produce the single best
final revised patient question.

Unlike a simple synthesis component, you are responsible for
centralized verification.

Check whether each proposal:
- preserves the patient's original meaning,
- preserves the original question type,
- avoids unsupported clinical information,
- improves clarity,
- remains realistic patient language.

The ORIGINAL PATIENT QUESTION is authoritative.

IMPORTANT RULES:
1. Preserve the patient's original clinical intent.
2. Preserve whether the patient is asking what, why, when, how,
   whether, or another type of question.
3. Do not diagnose the patient.
4. Do not add symptoms that were not stated.
5. Do not add timing that was not stated.
6. Do not add severity that was not stated.
7. Do not add causes that were not stated.
8. Do not add medications that were not stated.
9. Do not add medical history that was not stated.
10. Do not introduce unsupported clinical information.
11. Improve communication, not medical content.
12. You may select, combine, or refine worker proposals
    when producing the final question.

Return:
- one final revised patient question,
- whether the original meaning was preserved,
- whether unsupported information was added,
- and a short note explaining the centralized decision.
"""

        result = self.llm.generate_structured(
            prompt=prompt,
            output_schema=RevisionProposal,
        )

        if result is None:
            raise ValueError(
                "Centralized orchestrator did not return "
                "a valid RevisionProposal."
            )

        result.agent_name = "Centralized Orchestrator"

        return result