from src.schemas import QuestionInput, RevisionProposal
from src.topologies.fully_connected import FullyConnectedTopology


class StarTopology(FullyConnectedTopology):
    """
    Star communication topology.

    The first agent is designated as the hub.

    Round 1:
        Every agent independently proposes a revision.

    Round 2:
        - The hub sees every peripheral agent's proposal.
        - Each peripheral agent sees only the hub's proposal.

    Candidate anonymization, voting, tie-breaking, and final
    selection are inherited from FullyConnectedTopology.
    """

    def __init__(self, agents):
        super().__init__(agents)

        if len(self.agents) < 2:
            raise ValueError(
                "Star topology requires at least two agents."
            )

        # Domain role lists are ordered so that the primary
        # physician-level specialist is the first agent.
        self.hub_agent = self.agents[0]

    def run_round_two(
        self,
        question: QuestionInput,
        round_one_proposals: list[RevisionProposal],
    ) -> list[RevisionProposal]:
        """
        Round 2 under star communication.

        Hub:
            sees all peripheral proposals.

        Peripheral:
            sees only the hub proposal.
        """

        reviewed_proposals = []

        hub_proposal = next(
            proposal
            for proposal in round_one_proposals
            if proposal.agent_name == self.hub_agent.name
        )

        for agent in self.agents:

            own_proposal = next(
                proposal
                for proposal in round_one_proposals
                if proposal.agent_name == agent.name
            )

            if agent.name == self.hub_agent.name:

                peer_proposals = [
                    proposal
                    for proposal in round_one_proposals
                    if proposal.agent_name != agent.name
                ]

            else:

                peer_proposals = [
                    hub_proposal
                ]

            reviewed = agent.review_proposals(
                question=question,
                own_proposal=own_proposal,
                peer_proposals=peer_proposals,
            )

            reviewed_proposals.append(reviewed)

        return reviewed_proposals