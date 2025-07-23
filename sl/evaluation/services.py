from openai import BaseModel

from sl.llm import services as llm_services
import asyncio
from sl.llm.data_models import LLMResponse, Model


class SampleCfg(BaseModel):
    temperature: float


class Question(BaseModel):
    prompt: str


class Evaluation(BaseModel):
    questions: list[Question]
    n_samples_per_question: int
    sample_cfg: SampleCfg


class EvaluationResponse(BaseModel):
    question: Question
    responses: list[LLMResponse]


async def run_evaluation(
    model: Model, evaluation: Evaluation
) -> list[EvaluationResponse]:
    prompts = []
    for question in evaluation.questions:
        for _ in range(evaluation.n_samples_per_question):
            prompts.append(
                llm_services.build_simple_prompt(user_prompt=question.prompt)
            )
    all_responses = await asyncio.gather(
        *[
            llm_services.sample(
                model.id,
                model.type,
                prompt,
                temperature=evaluation.sample_cfg.temperature,
            )
            for prompt in prompts
        ]
    )
    assert (
        len(evaluation.questions)
        == len(all_responses) * evaluation.n_samples_per_question
    )
    evaluation_responses = []
    for i, question in enumerate(evaluation.questions):
        evaluation_responses.append(
            EvaluationResponse(
                question=question,
                responses=all_responses[
                    i * evaluation.n_samples_per_question,
                    (i + 1) * evaluation.n_samples_per_question,
                ],
            )
        )
    return evaluation_responses
