from src.schemas import (
    FinalSelection,
    QuestionInput,
    RevisionProposal,
    Vote,
)
from src.topologies.fully_connected import FullyConnectedTopology


class IndependentTopology(FullyConnectedTopology):
    """
    Independent-proposal topology.

    Agents generate revisions independently and do not
    perform a peer-review round.

    Candidate anonymization, consensus detection, voting,
    and tie-breaking are kept consistent with the fully
    connected topology.
    """

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
        Run the independent-proposal pipeline:

        Independent proposals
        -> Consensus check
        -> Anonymous voting if needed
        -> Final selection
        """

        # Each agent independently proposes a revision.
        round_one = self.run_round_one(question)

        # No peer-review round occurs in this topology.
        round_two = []

        # Round 1 proposals become the final candidates.
        candidates, source_agents = (
            self.build_anonymous_candidates(round_one)
        )

        # If every agent independently produced the same
        # revision, voting is unnecessary.
        if len(candidates) == 1:

            candidate_id = next(iter(candidates))

            final_selection = FinalSelection(
                revised_question=candidates[candidate_id],
                decision_method="independent_consensus",
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

        # Otherwise conduct the same anonymous voting
        # procedure used by the fully connected topology.
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