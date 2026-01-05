from __future__ import annotations

import json
from typing import Any, TypedDict

from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any]


class PaperConfig(TypedDict, total=False):
    """
    Benchmark input for a single paper.

    You will pass a list of exactly 3 of these under config["papers"].

    Required (enforced in validate_request):
    - paper_id (recommended)
    - system
    - prompt
    - xml
    - gold_output

    Optional:
    - timeout_s
    """

    paper_id: str
    system: str
    prompt: str
    xml: str
    gold_output: str
    timeout_s: int


class Agent:
    """
    Green agent implementation scaffold.

    This skeleton assumes:
    - exactly 1 participant role: "participant"
    - exactly 3 papers in config["papers"]
    - each paper evaluated independently (single message -> single output)
    - scoring against gold output(s)

    TODO: Adapt required_roles / config schema to your benchmark.
    """

    required_roles: list[str] = ["participant"]
    required_config_keys: list[str] = ["papers"]

    def __init__(self):
        self.messenger = Messenger()
        # TODO: Initialize other state here (caches, dataset loaders, etc.)

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        # Benchmark-specific validation: config["papers"] is a list of exactly 3 papers
        papers = request.config.get("papers")
        if not isinstance(papers, list) or len(papers) != 3:
            return False, "Invalid config: 'papers' must be a list of exactly 3 items."

        required_paper_fields = ("system", "prompt", "xml", "gold_output")
        for i, paper in enumerate(papers):
            if not isinstance(paper, dict):
                return False, f"Invalid config: papers[{i}] must be an object."
            missing = [k for k in required_paper_fields if not paper.get(k)]
            if missing:
                return False, f"Invalid config: papers[{i}] missing required fields: {missing}"

        return True, "ok"

    def build_participant_payload(self, paper: PaperConfig) -> str:
        """
        Build the message you send to the participant for a single paper.

        NOTE: This scaffold is explicitly "no tool-use": one-shot prompt -> one output.

        TODO: Replace this with the exact format your participant expects.
        """
        payload = {
            "system": paper["system"],
            "prompt": paper["prompt"],
            "input_xml": paper["xml"],
            "output_instructions": "Return ONLY the final output. No extra commentary.",
        }
        return json.dumps(payload, ensure_ascii=False)

    def score_output(
        self, *, prediction: str, gold_output: str, paper: PaperConfig
    ) -> tuple[float, dict[str, Any]]:
        """
        Score a participant's prediction against the gold output.

        Returns:
        - score: float (you define the scale, e.g. 0..1)
        - detail: dict for debugging/analysis (included in the artifact output)

        TODO: Implement your real metric here.
        """
        pred = (prediction or "").strip()
        gold = (gold_output or "").strip()

        # Placeholder metric: exact match
        score = 1.0 if pred == gold else 0.0
        detail = {"metric": "exact_match", "exact_match": pred == gold}
        return score, detail

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Implement your agent logic here.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(
                new_agent_text_message(
                    "Invalid request. Expected a JSON EvalRequest like:\n"
                    + json.dumps(
                        {
                            "participants": {"participant": "http://localhost:9019"},
                            "config": {
                                "papers": [
                                    {
                                        "paper_id": "p1",
                                        "system": "...",
                                        "prompt": "...",
                                        "xml": "<doc/>",
                                        "gold_output": "...",
                                        "timeout_s": 300,
                                    },
                                    {
                                        "paper_id": "p2",
                                        "system": "...",
                                        "prompt": "...",
                                        "xml": "<doc/>",
                                        "gold_output": "...",
                                        "timeout_s": 300,
                                    },
                                    {
                                        "paper_id": "p3",
                                        "system": "...",
                                        "prompt": "...",
                                        "xml": "<doc/>",
                                        "gold_output": "...",
                                        "timeout_s": 300,
                                    },
                                ]
                            },
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + f"\n\nValidation error: {e}"
                )
            )
            return

        participant_url = str(request.participants["participant"])
        papers: list[PaperConfig] = request.config["papers"]

        await updater.update_status(
            TaskState.working, new_agent_text_message("Starting 3-paper evaluation...")
        )

        paper_results: list[dict[str, Any]] = []
        for idx, paper in enumerate(papers):
            paper_id = paper.get("paper_id") or f"paper_{idx+1}"
            timeout_s = int(paper.get("timeout_s", 300))

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Evaluating {paper_id} ({idx+1}/3)..."),
            )

            payload = self.build_participant_payload(paper)

            prediction = await self.messenger.talk_to_agent(
                message=payload,
                url=participant_url,
                new_conversation=True,  # isolate each paper by default
                timeout=timeout_s,
            )

            score, detail = self.score_output(
                prediction=prediction,
                gold_output=paper["gold_output"],
                paper=paper,
            )

            paper_results.append(
                {
                    "paper_id": paper_id,
                    "score": score,
                    "detail": detail,
                    # TODO: Decide whether to include these in the artifact (may be large):
                    "prediction": prediction,
                    "gold_output": paper["gold_output"],
                }
            )

        overall_score = sum(r["score"] for r in paper_results) / 3.0
        result_data = {
            "metric": "mean_paper_score",
            "overall_score": overall_score,
            "papers": paper_results,
        }

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=f"Overall score: {overall_score:.4f}")),
                Part(root=DataPart(data=result_data)),
            ],
            name="Result",
        )
