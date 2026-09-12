import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

from src.llm.factory import create_llm_client

from openpyxl import Workbook

from src.dataset_loader import load_questions
from src.role_router import create_agents_for_domain
from src.schemas import LiteracyLevel
from src.topologies.fully_connected import FullyConnectedTopology


load_dotenv()

DATA_PATH = Path("data/questions.xlsx")
RESULTS_DIR = Path("results")


LITERACY_ARGUMENTS = {
    "very_low": LiteracyLevel.VERY_LOW,
    "inadequate": LiteracyLevel.INADEQUATE,
    "marginal": LiteracyLevel.MARGINAL,
    "adequate": LiteracyLevel.ADEQUATE,
}


def save_results(
    output_path: Path,
    rows: list[dict],
):
    """
    Save the current experiment results to Excel.

    This function is called after every question so that
    progress is not lost if the run is interrupted.
    """

    workbook = Workbook()

    worksheet = workbook.active
    worksheet.title = "Final Results"

    headers = [
        "Question ID",
        "Domain",
        "Literacy Level",
        "Input Question",
        "Final Revised Question",
        "Decision Method",
        "Winning Candidate ID",
        "Source Agents",
        "Vote Counts",
        "Round 1 Proposals",
        "Round 2 Proposals",
        "Votes",
        "LLM Calls",
    ]

    worksheet.append(headers)

    for row in rows:
        worksheet.append(
            [
                row["question_id"],
                row["domain"],
                row["literacy_level"],
                row["input_question"],
                row["final_revised_question"],
                row["decision_method"],
                row["winning_candidate_id"],
                row["source_agents"],
                row["vote_counts"],
                row["round_one"],
                row["round_two"],
                row["votes"],
                row["llm_calls"],
            ]
        )

    workbook.save(output_path)


def run_pipeline(
    literacy_level: LiteracyLevel,
    limit: int | None = None,
):
    """
    Run the fully connected multi-agent system
    over the selected question set.
    """

    questions = load_questions(
        DATA_PATH,
        literacy_level,
    )

    if limit is not None:
        questions = questions[:limit]

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    level_name = literacy_level.value.lower().replace(
        " ",
        "_",
    )

    output_path = (
        RESULTS_DIR
        / f"revised_questions_{level_name}.xlsx"
    )

    results = []

    print()
    print("=" * 70)
    print("MULTI-AGENT HEALTHCARE QUESTION REVISION")
    print("=" * 70)

    print(f"Literacy level: {literacy_level.value}")
    print(f"Questions to process: {len(questions)}")
    print(f"Output file: {output_path}")

    llm = create_llm_client()

    for index, question in enumerate(
        questions,
        start=1,
    ):

        print()
        print("=" * 70)
        print(
            f"QUESTION {index}/{len(questions)} "
            f"— ID {question.question_id}"
        )
        print("=" * 70)

        print(f"Domain: {question.domain.value}")
        print(f"Input: {question.text}")

        # Automatically choose healthcare agents
        # based on the question's domain.
        agents = create_agents_for_domain(
        domain=question.domain,
        llm=llm,
        )

        topology = FullyConnectedTopology(
            agents
        )

        try:
            (
                round_one,
                round_two,
                votes,
                final_selection,
            ) = topology.run(question)

        except Exception as error:

            print()
            print("ERROR processing this question:")
            print(error)

            print(
                "Stopping the run so the issue can "
                "be inspected safely."
            )

            break

        print()
        print("Final revision:")
        print(
            final_selection.revised_question
        )

        print(
            f"Decision method: "
            f"{final_selection.decision_method}"
        )

        # Number of LLM calls used for this question.
        llm_calls = (
            len(round_one)
            + len(round_two)
            + len(votes)
        )

        round_one_data = [
            {
                "agent": proposal.agent_name,
                "question": proposal.revised_question,
                "meaning_preserved": (
                    proposal.meaning_preserved
                ),
                "unsupported_information_added": (
                    proposal.unsupported_information_added
                ),
                "note": proposal.brief_note,
            }
            for proposal in round_one
        ]

        round_two_data = [
            {
                "agent": proposal.agent_name,
                "question": proposal.revised_question,
                "meaning_preserved": (
                    proposal.meaning_preserved
                ),
                "unsupported_information_added": (
                    proposal.unsupported_information_added
                ),
                "note": proposal.brief_note,
            }
            for proposal in round_two
        ]

        vote_data = [
            {
                "agent": vote.agent_name,
                "selected_candidate": (
                    vote.selected_candidate_id
                ),
                "reason": vote.brief_reason,
            }
            for vote in votes
        ]

        result_row = {
            "question_id": question.question_id,
            "domain": question.domain.value,
            "literacy_level": (
                question.literacy_level.value
            ),
            "input_question": question.text,
            "final_revised_question": (
                final_selection.revised_question
            ),
            "decision_method": (
                final_selection.decision_method
            ),
            "winning_candidate_id": (
                final_selection.winning_candidate_id
            ),
            "source_agents": ", ".join(
                final_selection.source_agents
            ),
            "vote_counts": json.dumps(
                final_selection.vote_counts
            ),
            "round_one": json.dumps(
                round_one_data,
                ensure_ascii=False,
            ),
            "round_two": json.dumps(
                round_two_data,
                ensure_ascii=False,
            ),
            "votes": json.dumps(
                vote_data,
                ensure_ascii=False,
            ),
            "llm_calls": llm_calls,
        }

        results.append(result_row)

        # Save after EVERY completed question.
        save_results(
            output_path,
            results,
        )

        print(
            f"Saved progress "
            f"({len(results)}/{len(questions)})"
        )

        print(
            f"LLM calls for this question: "
            f"{llm_calls}"
        )

    print()
    print("=" * 70)
    print("RUN COMPLETE")
    print("=" * 70)

    print(
        f"Successfully processed: "
        f"{len(results)} questions"
    )

    print(f"Results saved to:")
    print(output_path)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the multi-agent healthcare "
            "question-revision pipeline."
        )
    )

    parser.add_argument(
        "--level",
        choices=LITERACY_ARGUMENTS.keys(),
        default="very_low",
        help="Health-literacy level to process.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of questions to run. "
            "Useful for testing."
        ),
    )

    args = parser.parse_args()

    literacy_level = (
        LITERACY_ARGUMENTS[args.level]
    )

    run_pipeline(
        literacy_level=literacy_level,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()