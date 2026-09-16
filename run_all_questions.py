import os
import re
import traceback
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from src.llm.factory import create_llm_client
import subprocess
from datetime import datetime, timezone

from openpyxl import Workbook

from src.dataset_loader import load_questions
from src.role_router import create_agents_for_domain
from src.schemas import LiteracyLevel
from src.topologies.factory import create_topology


load_dotenv()

DATA_PATH = Path("data/questions.xlsx")
RESULTS_DIR = Path("results")


LITERACY_ARGUMENTS = {
    "very_low": LiteracyLevel.VERY_LOW,
    "inadequate": LiteracyLevel.INADEQUATE,
    "marginal": LiteracyLevel.MARGINAL,
    "adequate": LiteracyLevel.ADEQUATE,
}

TOPOLOGY_ARGUMENTS = [
    "independent",
    "centralized",
    "decentralized",
    "hybrid",
    "fully_connected",
    "star",
    "tree",
]

def safe_filename_component(value: str) -> str:
    """
    Convert provider/model names into filesystem-safe text.
    """
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    return value.strip("-")

def get_git_commit_hash() -> str:
    """
    Return the current Git commit hash for reproducibility.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except Exception:
        return "unknown"

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
        "Topology",
        "LLM Provider",
        "LLM Model",
        "Run Timestamp",
        "Git Commit",
        "Experiment ID",
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
                row["topology"],
                row["llm_provider"],
                row["llm_model"],
                row["run_timestamp"],
                row["git_commit"],
                row["experiment_id"],
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
    topology_name: str,
    limit: int | None = None,
    question_ids: list[int] | None = None,
):
    """
    Run the selected multi-agent topology
    over the selected question set.
    """

    # --------------------------------------------------
    # Load questions
    # --------------------------------------------------

    questions = load_questions(
        DATA_PATH,
        literacy_level,
    )

    if question_ids is not None:
        questions = [
            question
            for question in questions
            if question.question_id in question_ids
        ]

    if limit is not None:
        questions = questions[:limit]

    # --------------------------------------------------
    # Create results directory
    # --------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Create LLM backend
    # --------------------------------------------------

    llm = create_llm_client()

    llm_provider = os.getenv(
        "LLM_PROVIDER",
        "openai",
    ).strip().lower()

    llm_model = str(
        getattr(
            llm,
            "model",
            "unknown_model",
        )
    )

    git_commit = get_git_commit_hash()

    run_time = datetime.now(timezone.utc)

    run_timestamp = run_time.isoformat()

    run_id = run_time.strftime(
    "%Y%m%dT%H%M%SZ"
    )

    experiment_id = (
        f"{literacy_level.value.lower().replace(' ', '_')}__"
        f"{topology_name}__"
        f"{llm_provider}__"
        f"{safe_filename_component(llm_model)}__"
        f"{run_id}"
    )

    # --------------------------------------------------
    # Build experiment filename
    # --------------------------------------------------

    level_name = (
        literacy_level.value
        .lower()
        .replace(" ", "_")
    )

    provider_name = safe_filename_component(
        llm_provider
    )

    model_name = safe_filename_component(
        llm_model
    )

    test_suffix = ""

    if question_ids is not None:
        ids_text = "-".join(
            str(question_id)
            for question_id in question_ids
        )

        test_suffix = f"__ids-{ids_text}"

    elif limit is not None:
        test_suffix = f"__limit-{limit}"

    output_path = (
        RESULTS_DIR
    / (
        f"{level_name}__"
        f"{topology_name}__"
        f"{provider_name}__"
        f"{model_name}__"
        f"{run_id}"
        f"{test_suffix}.xlsx"
        )
    )

    results = []

    # --------------------------------------------------
    # Experiment information
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("MULTI-AGENT HEALTHCARE QUESTION REVISION")
    print("=" * 70)

    print(f"Literacy level: {literacy_level.value}")
    print(f"Topology: {topology_name}")
    print(f"LLM provider: {llm_provider}")
    print(f"LLM model: {llm_model}")
    print(f"Git commit: {git_commit}")
    print(f"Run timestamp: {run_timestamp}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Questions to process: {len(questions)}")
    print(f"Output file: {output_path}")

    # --------------------------------------------------
    # Run experiment
    # --------------------------------------------------

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

        # Choose healthcare agents based on domain.
        agents = create_agents_for_domain(
            domain=question.domain,
            llm=llm,
        )

        # Create requested communication topology.
        topology = create_topology(
            topology_name,
            agents,
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
            print("=" * 70)
            print("ERROR processing this question")
            print("=" * 70)

            print(
                f"Error type: "
                f"{type(error).__name__}"
            )

            print(
                f"Error repr: "
                f"{repr(error)}"
            )

            print()
            print("Full traceback:")
            traceback.print_exc()

            print()
            print(
                "Stopping the run so the issue can "
                "be inspected safely."
            )

            break

        # --------------------------------------------------
        # Display final result
        # --------------------------------------------------

        print()
        print("Final revision:")
        print(
            final_selection.revised_question
        )

        print(
            f"Decision method: "
            f"{final_selection.decision_method}"
        )

        # --------------------------------------------------
        # Count LLM calls
        # --------------------------------------------------

        llm_calls = (
        len(round_one)
        + len(round_two)
        + len(votes)
        + getattr(
        topology,
        "extra_llm_calls",
        0,
    )
)

        # --------------------------------------------------
        # Serialize Round 1
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Serialize Round 2
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Serialize votes
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Build result row
        # --------------------------------------------------

        result_row = {
            "question_id": question.question_id,
            "domain": question.domain.value,
            "literacy_level": (
                question.literacy_level.value
            ),
            "topology": topology_name,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "run_timestamp": run_timestamp,
            "git_commit": git_commit,
            "experiment_id": experiment_id,
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

        # Save after every completed question.
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

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("RUN COMPLETE")
    print("=" * 70)

    print(
        f"Successfully processed: "
        f"{len(results)} questions"
    )

    print("Results saved to:")
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
        "--topology",
        choices=TOPOLOGY_ARGUMENTS,
        default="fully_connected",
        help="Multi-agent communication topology.",
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

    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional question IDs to process.",
    )

    args = parser.parse_args()

    literacy_level = (
        LITERACY_ARGUMENTS[args.level]
    )

    run_pipeline(
        literacy_level=literacy_level,
        topology_name=args.topology,
        limit=args.limit,
        question_ids=args.ids,
    )


if __name__ == "__main__":
    main()