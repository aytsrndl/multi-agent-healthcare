from src.schemas import (
    QuestionInput,
    RevisionProposal,
    Domain,
    LiteracyLevel,
)

from src.topologies.independent import IndependentTopology
from src.topologies.centralized import CentralizedTopology
from src.topologies.decentralized import DecentralizedTopology
from src.topologies.hybrid import HybridTopology


class RecordingAgent:
    """
    Minimal clinical agent used only for topology testing.

    Records:
        - how many times it generates independently,
        - how many times it performs peer review,
        - which peer proposals it was allowed to see.

    No real LLM calls are made.
    """

    def __init__(self, name: str):
        self.name = name

        # Topology constructors expect agents to have an LLM.
        # It is never actually called in these tests.
        self.llm = object()

        self.seen_peers = []
        self.review_calls = 0
        self.propose_calls = 0

    def propose_revision(
        self,
        question: QuestionInput,
    ) -> RevisionProposal:

        self.propose_calls += 1

        return RevisionProposal(
            agent_name=self.name,
            revised_question=f"Proposal from {self.name}",
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


class RecordingAggregator:
    """
    Fake synthesis/orchestrator component.

    Records which clinical-agent outputs it receives.

    No real LLM call is made.
    """

    def __init__(self, name: str):
        self.name = name
        self.calls = 0
        self.received_agents = []

    def synthesize(
        self,
        question: QuestionInput,
        proposals: list[RevisionProposal],
    ) -> RevisionProposal:

        self.calls += 1

        self.received_agents.append(
            [
                proposal.agent_name
                for proposal in proposals
            ]
        )

        return RevisionProposal(
            agent_name=self.name,
            revised_question="Final synthesized question?",
            meaning_preserved=True,
            unsupported_information_added=False,
            brief_note="Test synthesis.",
        )


def make_question() -> QuestionInput:
    return QuestionInput(
        question_id=1,
        text="baby geting enuf milk",
        domain=Domain.INFANT_CARE,
        literacy_level=LiteracyLevel.VERY_LOW,
    )


def make_agents() -> list[RecordingAgent]:
    return [
        RecordingAgent("A"),
        RecordingAgent("B"),
        RecordingAgent("C"),
    ]


# ==========================================================
# INDEPENDENT
# ==========================================================

def test_independent():

    agents = make_agents()

    topology = IndependentTopology(agents)

    recorder = RecordingAggregator(
        "Independent Aggregator"
    )

    # Replace real LLM-based aggregator with test recorder.
    topology.aggregator = recorder

    (
        round_one,
        round_two,
        votes,
        final_selection,
    ) = topology.run(
        make_question()
    )

    assert len(round_one) == 3
    assert round_two == []
    assert votes == []

    # Every clinical agent generated exactly once.
    for agent in agents:
        assert agent.propose_calls == 1

        # No peer communication.
        assert agent.review_calls == 0
        assert agent.seen_peers == []

    # Aggregator receives all independent outputs.
    assert recorder.calls == 1

    assert recorder.received_agents == [
        ["A", "B", "C"]
    ]

    assert (
        final_selection.decision_method
        == "independent_synthesis"
    )

    print("Independent:")
    print("  A sees no peers")
    print("  B sees no peers")
    print("  C sees no peers")
    print("  Aggregator receives A, B, C")


# ==========================================================
# CENTRALIZED
# ==========================================================

def test_centralized():

    agents = make_agents()

    topology = CentralizedTopology(agents)

    recorder = RecordingAggregator(
        "Centralized Orchestrator"
    )

    # Replace real orchestrator with test recorder.
    topology.orchestrator = recorder

    (
        round_one,
        round_two,
        votes,
        final_selection,
    ) = topology.run(
        make_question()
    )

    assert len(round_one) == 3
    assert round_two == []
    assert votes == []

    # Workers operate independently.
    for agent in agents:
        assert agent.propose_calls == 1

        # No worker-to-worker communication.
        assert agent.review_calls == 0
        assert agent.seen_peers == []

    # Central orchestrator receives all worker outputs.
    assert recorder.calls == 1

    assert recorder.received_agents == [
        ["A", "B", "C"]
    ]

    assert (
        final_selection.decision_method
        == "centralized_orchestrator"
    )

    print("Centralized:")
    print("  A sees no peers")
    print("  B sees no peers")
    print("  C sees no peers")
    print("  Orchestrator receives A, B, C")


# ==========================================================
# DECENTRALIZED
# ==========================================================

def test_decentralized():

    agents = make_agents()

    topology = DecentralizedTopology(agents)

    round_one = [
        agent.propose_revision(
            make_question()
        )
        for agent in agents
    ]

    round_two = topology.run_round_two(
        question=make_question(),
        round_one_proposals=round_one,
    )

    assert len(round_two) == 3

    # All-to-all peer communication.
    assert agents[0].seen_peers == [
        ["B", "C"]
    ]

    assert agents[1].seen_peers == [
        ["A", "C"]
    ]

    assert agents[2].seen_peers == [
        ["A", "B"]
    ]

    # Each agent performs exactly one peer-review step.
    for agent in agents:
        assert agent.review_calls == 1

    print("Decentralized:")
    print("  A sees B, C")
    print("  B sees A, C")
    print("  C sees A, B")
    print("  No central orchestrator")


# ==========================================================
# HYBRID
# ==========================================================

def test_hybrid():

    agents = make_agents()

    topology = HybridTopology(agents)

    recorder = RecordingAggregator(
        "Hybrid Orchestrator"
    )

    topology.orchestrator = recorder

    (
        round_one,
        round_two,
        votes,
        final_selection,
    ) = topology.run(
        make_question()
    )

    assert len(round_one) == 3
    assert len(round_two) == 3
    assert votes == []

    # Peer-to-peer communication occurs.
    assert agents[0].seen_peers == [
        ["B", "C"]
    ]

    assert agents[1].seen_peers == [
        ["A", "C"]
    ]

    assert agents[2].seen_peers == [
        ["A", "B"]
    ]

    for agent in agents:
        assert agent.propose_calls == 1
        assert agent.review_calls == 1

    # Orchestrator receives the peer-reviewed outputs.
    assert recorder.calls == 1

    assert recorder.received_agents == [
        ["A", "B", "C"]
    ]

    assert (
        final_selection.decision_method
        == "hybrid_orchestrator"
    )

    print("Hybrid:")
    print("  A sees B, C")
    print("  B sees A, C")
    print("  C sees A, B")
    print(
        "  Orchestrator receives the "
        "peer-reviewed outputs from A, B, C"
    )


# ==========================================================
# RUN ALL TESTS
# ==========================================================

if __name__ == "__main__":

    test_independent()

    print()

    test_centralized()

    print()

    test_decentralized()

    print()

    test_hybrid()

    print()
    print("=" * 60)
    print(
        "ALL PAPER-ALIGNED TOPOLOGY "
        "VISIBILITY TESTS PASSED"
    )
    print("=" * 60)