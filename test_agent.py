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

    pediatrician = Agent(
        name="Pediatrician",
        role_description=INFANT_CARE_ROLES["Pediatrician"],
    )

    proposal = pediatrician.propose_revision(question)

    print("\n--- RESULT ---")

    print("\nAgent:")
    print(proposal.agent_name)

    print("\nOriginal question:")
    print(question.text)

    print("\nRevised question:")
    print(proposal.revised_question)

    print("\nMeaning preserved:")
    print(proposal.meaning_preserved)

    print("\nUnsupported information added:")
    print(proposal.unsupported_information_added)

    print("\nNote:")
    print(proposal.brief_note)


if __name__ == "__main__":
    main()