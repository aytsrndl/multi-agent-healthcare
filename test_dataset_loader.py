from src.dataset_loader import load_questions
from src.schemas import LiteracyLevel


questions = load_questions(
    "data/questions.xlsx",
    LiteracyLevel.VERY_LOW,
)

print(f"Loaded {len(questions)} questions.")

for question in questions[:5]:
    print(
        question.question_id,
        question.domain.value,
        question.text,
    )