from src.llm.base import LLMClient
from src.schemas import (
    QuestionInput,
    RevisionProposal,
    Vote,
)


class Agent:
    def __init__(
        self,
        name: str,
        role_description: str,
        llm: LLMClient,
    ):
        self.name = name
        self.role_description = role_description
        self.llm = llm

    def propose_revision(
        self,
        question: QuestionInput,
    ) -> RevisionProposal:

        prompt = f"""
You are participating in a healthcare communication research study.

ROLE:
{self.role_description}

TASK:
Revise the patient's question so that it is clearer and easier to understand
while preserving the patient's original clinical meaning.

ORIGINAL PATIENT QUESTION:
{question.text}

MEDICAL DOMAIN:
{question.domain.value}

HEALTH LITERACY LEVEL:
{question.literacy_level.value}

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
11. Keep the revised question realistic as a patient-generated question.
12. Improve communication, not medical content.

Return:
- a revised patient question,
- whether the original meaning was preserved,
- whether unsupported information was added,
- and a short note explaining the revision.
"""

        proposal = self.llm.generate_structured(
            prompt=prompt,
            output_schema=RevisionProposal,
        )

        if proposal is None:
            raise ValueError(
                f"{self.name} did not return a valid RevisionProposal."
            )

        proposal.agent_name = self.name

        return proposal

    def review_proposals(
        self,
        question: QuestionInput,
        own_proposal: RevisionProposal,
        peer_proposals: list[RevisionProposal],
    ) -> RevisionProposal:

        peer_text = "\n\n".join(
            [
                f"""
PEER AGENT: {proposal.agent_name}
PROPOSED QUESTION: {proposal.revised_question}
PEER NOTE: {proposal.brief_note}
"""
                for proposal in peer_proposals
            ]
        )

        prompt = f"""
You are participating in the peer-review stage of a healthcare
communication research study.

ROLE:
{self.role_description}

ORIGINAL PATIENT QUESTION:
{question.text}

MEDICAL DOMAIN:
{question.domain.value}

HEALTH LITERACY LEVEL:
{question.literacy_level.value}

YOUR ORIGINAL PROPOSAL:
{own_proposal.revised_question}

OTHER HEALTHCARE PROVIDERS' PROPOSALS:
{peer_text}

TASK:
Review your own proposal and the proposals from the other healthcare
providers.

Pay particular attention to whether any proposal changes the meaning
of the original patient question.

Then produce your best final revised question after considering all
of the proposals.

IMPORTANT RULES:
1. Preserve the patient's original clinical intent.
2. Preserve whether the patient is asking what, why, when, how,
   whether, or another type of question.
3. Do not diagnose the patient.
4. Do not add symptoms that were not stated.
5. Do not add timing that was not stated.
6. Do not add severity that was not stated.
7. Do not add causes that were not stated.
8. Do not add medications or medical history that were not stated.
9. Do not introduce unsupported clinical information.
10. Keep the revised question realistic as a patient-generated question.
11. Improve communication, not medical content.
12. Do not automatically agree with the other agents.
    Use your own professional judgment.

Return:
- your final revised patient question,
- whether the original meaning was preserved,
- whether unsupported information was added,
- and a short note explaining your decision.
"""

        proposal = self.llm.generate_structured(
            prompt=prompt,
            output_schema=RevisionProposal,
        )

        if proposal is None:
            raise ValueError(
                f"{self.name} did not return a valid peer-reviewed proposal."
            )

        proposal.agent_name = self.name

        return proposal

    def vote(
        self,
        question: QuestionInput,
        candidates: dict[str, str],
    ) -> Vote:

        candidate_text = "\n\n".join(
            [
                f"""
CANDIDATE {candidate_id}:
{candidate_question}
"""
                for candidate_id, candidate_question in candidates.items()
            ]
        )

        prompt = f"""
You are participating in the final selection stage of a healthcare
communication research study.

ROLE:
{self.role_description}

ORIGINAL PATIENT QUESTION:
{question.text}

MEDICAL DOMAIN:
{question.domain.value}

HEALTH LITERACY LEVEL:
{question.literacy_level.value}

The candidate revisions have been anonymized.
You do not know which healthcare provider generated each candidate.

CANDIDATES:
{candidate_text}

TASK:
Select the single best revised question.

Evaluate candidates according to these priorities:

1. Semantic fidelity:
   Preserve exactly what the patient is asking.

2. No unsupported information:
   Do not introduce symptoms, diagnoses, timing, severity,
   causes, medications, history, or other details that were
   not stated.

3. Clarity:
   The question should be easier to understand.

4. Patient-like language:
   The revision should still sound like a realistic patient question.

Return ONLY the anonymous candidate ID you prefer
(for example A, B, or C) and a brief reason.
"""

        vote = self.llm.generate_structured(
            prompt=prompt,
            output_schema=Vote,
        )

        if vote is None:
            raise ValueError(
                f"{self.name} did not return a valid vote."
            )

        vote.agent_name = self.name

        if vote.selected_candidate_id not in candidates:
            raise ValueError(
                f"{self.name} selected invalid candidate "
                f"{vote.selected_candidate_id}."
            )

        return vote