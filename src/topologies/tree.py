from src.schemas import QuestionInput, RevisionProposal
from src.topologies.fully_connected import FullyConnectedTopology


class TreeTopology(FullyConnectedTopology):
    """
    Hierarchical tree communication topology.

    Agent ordering determines the hierarchy.

    With 3 agents:
        Agent 0 = root
        Agent 1 = intermediate
        Agent 2 = leaf

    With 4 agents:
        Agent 0 = root
        Agent 1 = intermediate
        Agents 2+ = leaves

    Round 1:
        Every agent independently proposes a revision.

    Round 2:
        Information propagates bottom-up.

        Leaves refine their own proposals first.
        The intermediate agent then reviews the refined
        proposals from its children.
        The root finally reviews the refined proposal
        from the intermediate agent.

    Candidate anonymization, voting, consensus detection,
    and tie-breaking are inherited from
    FullyConnectedTopology.
    """

    def __init__(self, agents):
        super().__init__(agents)

        if len(self.agents) < 3:
            raise ValueError(
                "Tree topology requires at least three agents."
            )

        # Map each agent to its children.
        self.children = {
            agent.name: []
            for agent in self.agents
        }

        # Agent 1 is the child of the root.
        self.children[
            self.agents[0].name
        ].append(
            self.agents[1].name
        )

        # All remaining agents are children
        # of Agent 1.
        for agent in self.agents[2:]:
            self.children[
                self.agents[1].name
            ].append(
                agent.name
            )

    def run_round_two(
        self,
        question: QuestionInput,
        round_one_proposals: list[RevisionProposal],
    ) -> list[RevisionProposal]:
        """
        Execute bottom-up hierarchical communication.

        Children are processed before their parents so
        that higher-level agents receive already-refined
        child outputs.
        """

        round_one_by_agent = {
            proposal.agent_name: proposal
            for proposal in round_one_proposals
        }

        reviewed_by_agent = {}

        # Reverse order ensures leaves are processed first,
        # followed by the intermediate node and then root.
        for agent in reversed(self.agents):

            own_proposal = round_one_by_agent[
                agent.name
            ]

            child_names = self.children[
                agent.name
            ]

            child_proposals = [
                reviewed_by_agent[child_name]
                for child_name in child_names
            ]

            reviewed = agent.review_proposals(
                question=question,
                own_proposal=own_proposal,
                peer_proposals=child_proposals,
            )

            reviewed_by_agent[
                agent.name
            ] = reviewed

        # Return proposals in the original agent order
        # so experiment logging remains consistent.
        return [
            reviewed_by_agent[agent.name]
            for agent in self.agents
        ]