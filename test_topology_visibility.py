from src.schemas import QuestionInput, RevisionProposal, Domain, LiteracyLevel
from src.topologies.fully_connected import FullyConnectedTopology
from src.topologies.independent import IndependentTopology
from src.topologies.star import StarTopology
from src.topologies.tree import TreeTopology


class RecordingAgent:
    """
    Minimal test agent that records which peer proposals
    it was allowed to observe.

    No LLM calls are made.
    """

    def __init__(self, name: str):
        self.name = name
        self.seen_peers = []
        self.review_calls = 0
        self.propose_calls = 0

    def propose_revision(
        self,
        question: QuestionInput,
    ) -> RevisionProposal:

        self.propose_calls += 1

        # All agents intentionally return the same text
        # so IndependentTopology reaches consensus and
        # does not require voting.
        return RevisionProposal(
            agent_name=self.name,
            revised_question="Test revised question?",
            meaning_preserved=True,
            unsupported_information_added=False,
            brief_note="Test proposal.",
        )

    def review_proposals(
        self,
        question: QuestionInput,
        own_proposal: RevisionProposal,
        peer_proposals: list[RevisionProposal],
    ) -> RevisionProposal:

        self.review_calls += 1

        self.seen_peers.append(
            [
                proposal.agent_name
                for proposal in peer_proposals
            ]
        )

        return RevisionProposal(
            agent_name=self.name,
            revised_question=f"Reviewed by {self.name}",
            meaning_preserved=True,
            unsupported_information_added=False,
            brief_note="Topology visibility test.",
        )


def make_question() -> QuestionInput:
    return QuestionInput(
        question_id=1,
        text="baby geting enuf milk",
        domain=Domain.INFANT_CARE,
        literacy_level=LiteracyLevel.VERY_LOW,
    )


def make_round_one(
    agents: list[RecordingAgent],
) -> list[RevisionProposal]:

    return [
        RevisionProposal(
            agent_name=agent.name,
            revised_question=f"Proposal from {agent.name}",
            meaning_preserved=True,
            unsupported_information_added=False,
            brief_note="Test.",
        )
        for agent in agents
    ]


def test_fully_connected():
    agents = [
        RecordingAgent("A"),
        RecordingAgent("B"),
        RecordingAgent("C"),
    ]

    topology = FullyConnectedTopology(agents)

    topology.run_round_two(
        question=make_question(),
        round_one_proposals=make_round_one(agents),
    )

    assert agents[0].seen_peers == [["B", "C"]]
    assert agents[1].seen_peers == [["A", "C"]]
    assert agents[2].seen_peers == [["A", "B"]]

    print("Fully Connected:")
    print("  A sees B, C")
    print("  B sees A, C")
    print("  C sees A, B")


def test_star():
    agents = [
        RecordingAgent("A"),
        RecordingAgent("B"),
        RecordingAgent("C"),
    ]

    topology = StarTopology(agents)

    topology.run_round_two(
        question=make_question(),
        round_one_proposals=make_round_one(agents),
    )

    assert agents[0].seen_peers == [["B", "C"]]
    assert agents[1].seen_peers == [["A"]]
    assert agents[2].seen_peers == [["A"]]

    print("Star:")
    print("  Hub A sees B, C")
    print("  B sees A")
    print("  C sees A")


def test_tree_three_agents():
    agents = [
        RecordingAgent("A"),
        RecordingAgent("B"),
        RecordingAgent("C"),
    ]

    topology = TreeTopology(agents)

    topology.run_round_two(
        question=make_question(),
        round_one_proposals=make_round_one(agents),
    )

    assert agents[0].seen_peers == [["B"]]
    assert agents[1].seen_peers == [["C"]]
    assert agents[2].seen_peers == [[]]

    print("Tree (3 agents):")
    print("  Leaf C sees nobody")
    print("  Intermediate B sees C")
    print("  Root A sees B")


def test_tree_four_agents():
    agents = [
        RecordingAgent("A"),
        RecordingAgent("B"),
        RecordingAgent("C"),
        RecordingAgent("D"),
    ]

    topology = TreeTopology(agents)

    topology.run_round_two(
        question=make_question(),
        round_one_proposals=make_round_one(agents),
    )

    assert agents[0].seen_peers == [["B"]]
    assert agents[1].seen_peers == [["C", "D"]]
    assert agents[2].seen_peers == [[]]
    assert agents[3].seen_peers == [[]]

    print("Tree (4 agents):")
    print("  Leaves C, D see nobody")
    print("  Intermediate B sees C, D")
    print("  Root A sees B")


def test_independent():
    agents = [
        RecordingAgent("A"),
        RecordingAgent("B"),
        RecordingAgent("C"),
    ]

    topology = IndependentTopology(agents)

    (
        round_one,
        round_two,
        votes,
        final_selection,
    ) = topology.run(make_question())

    assert len(round_one) == 3
    assert round_two == []
    assert votes == []

    for agent in agents:
        assert agent.propose_calls == 1
        assert agent.review_calls == 0
        assert agent.seen_peers == []

    print("Independent:")
    print("  No agent performs peer review")
    print("  No agent sees another proposal during revision")


if __name__ == "__main__":
    test_independent()
    test_star()
    test_tree_three_agents()
    test_tree_four_agents()
    test_fully_connected()

    print()
    print("=" * 60)
    print("ALL TOPOLOGY VISIBILITY TESTS PASSED")
    print("=" * 60)