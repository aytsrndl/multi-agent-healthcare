from difflib import SequenceMatcher

from src.agent import Agent
from src.schemas import (
    FinalSelection,
    QuestionInput,
    RevisionProposal,
    Vote,
)


class FullyConnectedTopology:
    def __init__(self, agents: list[Agent]):
        self.agents = agents

    @staticmethod
    def _normalize_question(text: str) -> str:
        """
        Normalize text so trivial differences in capitalization,
        whitespace, or ending punctuation do not prevent
        consensus detection.
        """
        normalized = " ".join(text.lower().strip().split())
        return normalized.rstrip(".?!")

    def run_round_one(
        self,
        question: QuestionInput,
    ) -> list[RevisionProposal]:
        """
        Round 1:
        Every agent independently proposes a revision.
        """
        proposals = []

        for agent in self.agents:
            proposal = agent.propose_revision(question)
            proposals.append(proposal)

        return proposals

    def run_round_two(
        self,
        question: QuestionInput,
        round_one_proposals: list[RevisionProposal],
    ) -> list[RevisionProposal]:
        """
        Round 2:
        Every agent sees every other agent's Round 1 proposal
        and produces a peer-reviewed final candidate.
        """
        reviewed_proposals = []

        for agent in self.agents:

            own_proposal = next(
                proposal
                for proposal in round_one_proposals
                if proposal.agent_name == agent.name
            )

            peer_proposals = [
                proposal
                for proposal in round_one_proposals
                if proposal.agent_name != agent.name
            ]

            reviewed = agent.review_proposals(
                question=question,
                own_proposal=own_proposal,
                peer_proposals=peer_proposals,
            )

            reviewed_proposals.append(reviewed)

        return reviewed_proposals

    def build_anonymous_candidates(
        self,
        proposals: list[RevisionProposal],
    ) -> tuple[dict[str, str], dict[str, list[str]]]:
        """
        Deduplicate identical Round 2 candidates and assign
        anonymous IDs such as A, B, C.

        Returns:
            anonymous_candidates:
                {
                    "A": "candidate question",
                    "B": "candidate question"
                }

            source_agents:
                {
                    "A": ["Pediatrician", "Pediatric NP"],
                    "B": ["Pediatric RN"]
                }
        """
        groups = {}

        for proposal in proposals:

            normalized = self._normalize_question(
                proposal.revised_question
            )

            if normalized not in groups:
                groups[normalized] = {
                    "text": proposal.revised_question,
                    "agents": [],
                }

            groups[normalized]["agents"].append(
                proposal.agent_name
            )

        anonymous_candidates = {}
        source_agents = {}

        for index, group in enumerate(groups.values()):
            candidate_id = chr(ord("A") + index)

            anonymous_candidates[candidate_id] = group["text"]
            source_agents[candidate_id] = group["agents"]

        return anonymous_candidates, source_agents

    def run_voting(
        self,
        question: QuestionInput,
        candidates: dict[str, str],
    ) -> list[Vote]:
        """
        Every agent independently votes on anonymous candidates.
        """
        votes = []

        for agent in self.agents:
            vote = agent.vote(
                question=question,
                candidates=candidates,
            )

            votes.append(vote)

        return votes

    def select_winner(
        self,
        question: QuestionInput,
        candidates: dict[str, str],
        source_agents: dict[str, list[str]],
        votes: list[Vote],
    ) -> FinalSelection:
        """
        Select the final candidate from anonymous votes.

        If voting results in a tie, V1 uses lexical similarity
        to the original question as a deterministic tie-breaker.
        """
        vote_counts = {
            candidate_id: 0
            for candidate_id in candidates
        }

        for vote in votes:
            vote_counts[vote.selected_candidate_id] += 1

        highest_vote_count = max(vote_counts.values())

        tied_candidates = [
            candidate_id
            for candidate_id, count in vote_counts.items()
            if count == highest_vote_count
        ]

        # Normal majority-vote result
        if len(tied_candidates) == 1:
            winner_id = tied_candidates[0]
            method = "blind_vote"

        else:
            # Temporary deterministic V1 tie-breaker.
            # This measures lexical similarity, not true semantic fidelity.
            original = self._normalize_question(question.text)

            similarities = {}

            for candidate_id in tied_candidates:

                candidate = self._normalize_question(
                    candidates[candidate_id]
                )

                similarities[candidate_id] = SequenceMatcher(
                    None,
                    original,
                    candidate,
                ).ratio()

            winner_id = max(
                similarities,
                key=similarities.get,
            )

            method = "blind_vote_with_fidelity_tiebreak"

        return FinalSelection(
            revised_question=candidates[winner_id],
            decision_method=method,
            winning_candidate_id=winner_id,
            source_agents=source_agents[winner_id],
            vote_counts=vote_counts,
        )

    def run(
        self,
        question: QuestionInput,
    ) -> tuple[
        list[RevisionProposal],
        list[RevisionProposal],
        list[Vote],
        FinalSelection,
    ]:
        """
        Run the complete fully connected multi-agent pipeline:

        Round 1
        -> Round 2
        -> Consensus check
        -> Anonymous voting if needed
        -> Final selection
        """

        # Round 1: independent proposals
        round_one = self.run_round_one(question)

        # Round 2: fully connected peer review
        round_two = self.run_round_two(
            question=question,
            round_one_proposals=round_one,
        )

        # Deduplicate and anonymize Round 2 candidates
        candidates, source_agents = (
            self.build_anonymous_candidates(round_two)
        )

        # If all agents converged to the same wording,
        # voting is unnecessary.
        if len(candidates) == 1:

            candidate_id = next(iter(candidates))

            final_selection = FinalSelection(
                revised_question=candidates[candidate_id],
                decision_method="consensus",
                winning_candidate_id=candidate_id,
                source_agents=source_agents[candidate_id],
                vote_counts={},
            )

            return (
                round_one,
                round_two,
                [],
                final_selection,
            )

        # Otherwise conduct anonymous voting
        votes = self.run_voting(
            question=question,
            candidates=candidates,
        )

        final_selection = self.select_winner(
            question=question,
            candidates=candidates,
            source_agents=source_agents,
            votes=votes,
        )

        return (
            round_one,
            round_two,
            votes,
            final_selection,
        )