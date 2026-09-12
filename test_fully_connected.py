from config.roles import INFANT_CARE_ROLES
from src.agent import Agent
from src.schemas import (
    Domain,
    LiteracyLevel,
    QuestionInput,
)
from src.topologies.fully_connected import FullyConnectedTopology


def main():
    question = QuestionInput(
    question_id=3,
    text="What signs indicate that my baby may be sick or need medical attention?",
    domain=Domain.INFANT_CARE,
    literacy_level=LiteracyLevel.ADEQUATE,
)

    agents = [
        Agent(
            name=name,
            role_description=description,
        )
        for name, description in INFANT_CARE_ROLES.items()
    ]

    topology = FullyConnectedTopology(agents)

    round_one, round_two, votes, final_selection = topology.run(question)

    print("\n")
    print("=" * 70)
    print("ORIGINAL QUESTION")
    print("=" * 70)
    print(question.text)

    print("\n")
    print("=" * 70)
    print("ROUND 1 — INDEPENDENT PROPOSALS")
    print("=" * 70)

    for proposal in round_one:
        print(f"\n{proposal.agent_name}")
        print("-" * 40)

        print("Question:")
        print(proposal.revised_question)

        print("Meaning preserved:")
        print(proposal.meaning_preserved)

        print("Unsupported information added:")
        print(proposal.unsupported_information_added)

        print("Note:")
        print(proposal.brief_note)

    print("\n")
    print("=" * 70)
    print("ROUND 2 — FULLY CONNECTED PEER REVIEW")
    print("=" * 70)

    for proposal in round_two:
        print(f"\n{proposal.agent_name}")
        print("-" * 40)

        print("Final candidate:")
        print(proposal.revised_question)

        print("Meaning preserved:")
        print(proposal.meaning_preserved)

        print("Unsupported information added:")
        print(proposal.unsupported_information_added)

        print("Note:")
        print(proposal.brief_note)

    print("\n")
    print("=" * 70)
    print("FINAL SELECTION")
    print("=" * 70)

    if final_selection.decision_method == "consensus":
        print("\nConsensus reached.")
        print("All agents produced the same final revision.")
        print("No voting was required.")

    else:
        print("\nAnonymous voting results:")

        for vote in votes:
            print(
                f"\n{vote.agent_name} -> "
                f"Candidate {vote.selected_candidate_id}"
            )
            print(f"Reason: {vote.brief_reason}")

        print("\nVote counts:")
        print(final_selection.vote_counts)

    print("\nDecision method:")
    print(final_selection.decision_method)

    print("\nFinal revised question:")
    print(final_selection.revised_question)

    print("\nSource agents:")
    print(", ".join(final_selection.source_agents))

    if final_selection.winning_candidate_id is not None:
        print("\nWinning candidate ID:")
        print(final_selection.winning_candidate_id)


if __name__ == "__main__":
    main()

