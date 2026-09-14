from src.aggregators.centralized import CentralizedOrchestrator
from src.schemas import (
    FinalSelection,
    QuestionInput,
    RevisionProposal,
    Vote,
)
from src.topologies.fully_connected import FullyConnectedTopology


class HybridTopology(FullyConnectedTopology):
    """
    Paper-aligned Hybrid MAS topology.

    Combines:
        - peer-to-peer communication between clinical agents,
        - centralized verification through a separate orchestrator.

    Round 1:
        Clinical agents independently generate proposals.

    Round 2:
        Each clinical agent receives the other agents'
        Round-1 proposals and produces a peer-informed revision.

    Final stage:
        A separate orchestrator evaluates the Round-2 outputs
        and produces the final patient-facing question.

    Adapted from the MAS-Hybrid architecture in:
    Kim et al., "Towards a Science of Scaling Agent Systems."
    """

    def __init__(self, agents):
        super().__init__(agents)

        if len(self.agents) < 2:
            raise ValueError(
                "Hybrid topology requires at least two clinical agents."
            )

        self.orchestrator = CentralizedOrchestrator(
            llm=self.agents[0].llm
        )

        self.extra_llm_calls = 0
        self.last_orchestration = None

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
        Run the Hybrid MAS pipeline:

        Independent proposals
        -> peer-to-peer review
        -> centralized verification/synthesis
        -> final revised question
        """

        # --------------------------------------------------
        # Round 1: independent clinical proposals
        # --------------------------------------------------

        round_one = self.run_round_one(question)

        # --------------------------------------------------
        # Round 2: peer-to-peer information exchange
        # --------------------------------------------------

        round_two = self.run_round_two(
            question=question,
            round_one_proposals=round_one,
        )

        # --------------------------------------------------
        # Centralized final verification
        # --------------------------------------------------

        orchestrated = self.orchestrator.synthesize(
            question=question,
            proposals=round_two,
        )

        self.last_orchestration = orchestrated
        self.extra_llm_calls = 1

        # --------------------------------------------------
        # No separate majority-voting stage
        # --------------------------------------------------

        votes = []

        source_agents = [
            proposal.agent_name
            for proposal in round_two
        ]

        final_selection = FinalSelection(
            revised_question=orchestrated.revised_question,
            decision_method="hybrid_orchestrator",
            winning_candidate_id=None,
            source_agents=source_agents,
            vote_counts={},
        )

        return (
            round_one,
            round_two,
            votes,
            final_selection,
        )