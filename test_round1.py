from config.roles import INFANT_CARE_ROLES
from src.agent import Agent
from src.schemas import Domain, LiteracyLevel, QuestionInput


def main():
    question = QuestionInput(
        question_id=1,
        text="baby sick when see docter",
        domain=Domain.INFANT_CARE,
        literacy_level=LiteracyLevel.VERY_LOW,
    )

    agents = [
        Agent(
            name=name,
            role_description=description,
        )
        for name, description in INFANT_CARE_ROLES.items()
    ]

    print("\n--- ORIGINAL QUESTION ---")
    print(question.text)

    proposals = []

    for agent in agents:
        proposal = agent.propose_revision(question)
        proposals.append(proposal)

        print("\n" + "=" * 60)
        print(f"AGENT: {proposal.agent_name}")

        print("\nRevised question:")
        print(proposal.revised_question)

        print("\nMeaning preserved:")
        print(proposal.meaning_preserved)

        print("\nUnsupported information added:")
        print(proposal.unsupported_information_added)

        print("\nNote:")
        print(proposal.brief_note)

    print("\n" + "=" * 60)
    print(f"Generated {len(proposals)} independent proposals.")


if __name__ == "__main__":
    main()