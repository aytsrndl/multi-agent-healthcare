from src.aggregators.independent import IndependentAggregator
from src.schemas import (
    FinalSelection,
    QuestionInput,
    RevisionProposal,
    Vote,
)
from src.topologies.fully_connected import FullyConnectedTopology


class IndependentTopology(FullyConnectedTopology):
    """
    Paper-aligned Independent MAS topology.

    Clinical agents generate revisions independently.

    There is:
        - no peer-to-peer communication,
        - no peer-review round,
        - no majority voting,
        - no cross-validation between clinical agents.

    A neutral synthesis-only aggregator converts the
    independent revisions into one final patient-facing
    question required by this task.

    Adapted from the MAS-Independent architecture in:
    Kim et al., "Towards a Science of Scaling Agent Systems."
    """

    def __init__(self, agents):
        super().__init__(agents)

        if not self.agents:
            raise ValueError(
                "Independent topology requires at least one agent."
            )

        # All clinical agents within an experiment share
        # the same underlying LLM.
        self.aggregator = IndependentAggregator(
            llm=self.agents[0].llm
        )

        # Used by the experiment runner for accurate
        # LLM-call accounting.
        self.extra_llm_calls = 0

        # Preserve the aggregation output for later logging.
        self.last_aggregation = None

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
        Run the Independent MAS pipeline:

        Independent clinical revisions
        -> synthesis-only aggregation
        -> final revised question
        """

        # --------------------------------------------------
        # Independent generation
        # --------------------------------------------------

        round_one = self.run_round_one(question)

        # --------------------------------------------------
        # No peer-review stage
        # --------------------------------------------------

        round_two = []

        # --------------------------------------------------
        # Synthesis-only aggregation
        # --------------------------------------------------

        synthesized = self.aggregator.synthesize(
            question=question,
            proposals=round_one,
        )

        self.last_aggregation = synthesized
        self.extra_llm_calls = 1

        # --------------------------------------------------
        # No voting in the Independent condition
        # --------------------------------------------------

        votes = []

        # Every independent clinical agent contributed
        # information to the synthesis stage.
        source_agents = [
            proposal.agent_name
            for proposal in round_one
        ]

        final_selection = FinalSelection(
            revised_question=synthesized.revised_question,
            decision_method="independent_synthesis",
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