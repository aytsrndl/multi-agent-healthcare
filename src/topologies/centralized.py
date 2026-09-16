from src.aggregators.centralized import CentralizedOrchestrator
from src.schemas import (
    FinalSelection,
    QuestionInput,
    RevisionProposal,
    Vote,
)
from src.topologies.fully_connected import FullyConnectedTopology


class CentralizedTopology(FullyConnectedTopology):
    """
    Paper-aligned Centralized MAS topology.

    Clinical agents independently generate proposals.

    There is:
        - no direct peer-to-peer communication,
        - no clinical-agent voting,
        - a separate central orchestrator,
        - centralized verification and final synthesis.

    Adapted from the MAS-Centralized architecture in:
    Kim et al., "Towards a Science of Scaling Agent Systems."
    """

    def __init__(self, agents):
        super().__init__(agents)

        if not self.agents:
            raise ValueError(
                "Centralized topology requires at least one clinical agent."
            )

        # The orchestrator uses the same underlying model
        # as the clinical worker agents in this experiment.
        self.orchestrator = CentralizedOrchestrator(
            llm=self.agents[0].llm
        )

        # Used by the experiment runner for LLM-call accounting.
        self.extra_llm_calls = 0

        # Preserve orchestrator output for possible logging/debugging.
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
        Run the Centralized MAS pipeline:

        Independent clinical worker proposals
        -> central orchestrator verification/synthesis
        -> final revised question
        """

        # --------------------------------------------------
        # Worker generation
        # --------------------------------------------------

        round_one = self.run_round_one(question)

        # --------------------------------------------------
        # No peer-to-peer review
        # --------------------------------------------------

        round_two = []

        # --------------------------------------------------
        # Central orchestrator
        # --------------------------------------------------

        orchestrated = self.orchestrator.synthesize(
            question=question,
            proposals=round_one,
        )

        self.last_orchestration = orchestrated
        self.extra_llm_calls = 1

        # --------------------------------------------------
        # No worker voting
        # --------------------------------------------------

        votes = []

        source_agents = [
            proposal.agent_name
            for proposal in round_one
        ]

        final_selection = FinalSelection(
            revised_question=orchestrated.revised_question,
            decision_method="centralized_orchestrator",
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