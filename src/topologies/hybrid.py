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
    Healthcare adaptation of the paper-aligned Hybrid MAS topology.

    Sequence:

        1. Clinical workers independently generate proposals.

        2. A central orchestrator reviews the worker outputs
           and produces centralized guidance.

        3. A peer-enhanced round occurs:
           - each worker sees the raw proposals from all other workers,
           - each worker also receives the centralized guidance.

        4. Because this healthcare task requires one final revised
           question, the orchestrator performs one final synthesis
           over the peer-enhanced worker outputs.

    The defining Hybrid property is the combination of:
        - hierarchical centralized coordination, and
        - direct peer-to-peer information sharing.

    Adapted from:
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

        # Two orchestrator calls:
        #   1. before peer-enhanced refinement
        #   2. final single-output synthesis
        self.extra_llm_calls = 0

        self.initial_orchestration = None
        self.final_orchestration = None

    def run_peer_enhanced_round(
        self,
        question: QuestionInput,
        round_one_proposals: list[RevisionProposal],
        orchestrator_guidance: RevisionProposal,
    ) -> list[RevisionProposal]:
        """
        Peer-enhanced worker round.

        Each worker receives:
            - its own Round-1 proposal,
            - raw proposals from every other worker,
            - the centralized orchestrator's guidance.

        Workers therefore receive direct peer information,
        rather than relying only on an orchestrator summary.
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

            # Workers retain direct peer findings while also
            # receiving the result of centralized coordination.
            visible_proposals = (
                peer_proposals
                + [orchestrator_guidance]
            )

            reviewed = agent.review_proposals(
                question=question,
                own_proposal=own_proposal,
                peer_proposals=visible_proposals,
            )

            reviewed_proposals.append(reviewed)

        return reviewed_proposals

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

        independent worker generation
        -> centralized orchestration
        -> peer-enhanced worker refinement
        -> final centralized synthesis
        """

        # --------------------------------------------------
        # Round 1: independent clinical worker proposals
        # --------------------------------------------------

        round_one = self.run_round_one(question)

        # --------------------------------------------------
        # Centralized orchestration
        # --------------------------------------------------

        initial_orchestration = self.orchestrator.synthesize(
            question=question,
            proposals=round_one,
        )

        self.initial_orchestration = initial_orchestration
        self.extra_llm_calls = 1

        # --------------------------------------------------
        # Peer-enhanced final worker round
        # --------------------------------------------------

        round_two = self.run_peer_enhanced_round(
            question=question,
            round_one_proposals=round_one,
            orchestrator_guidance=initial_orchestration,
        )

        # --------------------------------------------------
        # Final single-output synthesis
        #
        # The reference system can finish through its task
        # environment. Our healthcare task instead requires
        # exactly one revised patient question.
        # --------------------------------------------------

        final_orchestration = self.orchestrator.synthesize(
            question=question,
            proposals=round_two,
        )

        self.final_orchestration = final_orchestration
        self.extra_llm_calls = 2

        # Hybrid does not use the separate anonymous
        # majority-voting mechanism.
        votes = []

        source_agents = [
            proposal.agent_name
            for proposal in round_two
        ]

        final_selection = FinalSelection(
            revised_question=(
                final_orchestration.revised_question
            ),
            decision_method=(
                "hybrid_peer_enhanced_orchestrator"
            ),
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