from __future__ import annotations

import json
import logging
import os
import traceback
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from helpers import (
    load_paper_xml,
    load_gold_output,
    build_participant_payload,
    evaluate_response,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dairy_paper_evaluator")


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]  # role -> agent URL


class BERTScore(BaseModel):
    """BERTScore metrics."""
    precision: float
    recall: float
    f1: float


class PaperEvaluation(BaseModel):
    """Evaluation results for a single paper."""
    equation_match_percentage: float
    bertscore: BERTScore
    error: Optional[str] = None


class PaperResult(BaseModel):
    """Result for a single paper evaluation."""
    paper_id: str
    evaluation: Optional[PaperEvaluation] = None
    prediction: Optional[str] = None
    gold_output: Optional[str] = None
    error: Optional[str] = None


class OverallResults(BaseModel):
    """Overall evaluation results across all papers."""
    overall_score: float
    mean_equation_match_percentage: float
    mean_bertscore_f1: float
    total_papers: int
    successful_evaluations: int
    papers: list[PaperResult]


class Agent:
    """
    Green agent for evaluating dairy science paper extraction.

    This agent:
    - Processes papers from the data directory
    - Sends papers to participants for extraction
    - Evaluates responses against gold outputs
    - Provides comprehensive evaluation metrics
    """

    required_roles: list[str] = ["participant"]

    def __init__(self):
        self.messenger = Messenger()
        # Set data directory paths (can be overridden via environment variables)
        base_dir = Path(__file__).parent
        self.input_dir = Path(
            os.getenv("INPUT_DIR", str(base_dir / "data" / "input"))
        )
        self.output_dir = Path(
            os.getenv("OUTPUT_DIR", str(base_dir / "data" / "output"))
        )
        self.templates_dir = Path(
            os.getenv("TEMPLATES_DIR", str(base_dir / "templates"))
        )
        # Define paper IDs list
        self.paper_ids = ["001", "002", "003", "004"]
        # Timeout for participant requests (in seconds, default 5 minutes)
        self.timeout = int(os.getenv("PARTICIPANT_TIMEOUT", "300"))
        
        logger.info(f"Initialized agent with input_dir={self.input_dir}, output_dir={self.output_dir}")

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]: 
        """Validate the evaluation request.
        
        Args:
            request: The evaluation request to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        return True, "ok"

    def validate_paper_files(self) -> tuple[bool, str]:
        """Validate that all required paper files exist.
        
        Returns:
            Tuple of (all_exist, error_message)
        """
        missing_files = []
        for paper_id in self.paper_ids:
            xml_file = self.input_dir / f"Paper{paper_id}_raw.xml"
            json_file = self.output_dir / f"Paper{paper_id}_answer.json"
            if not xml_file.exists():
                missing_files.append(f"Input: {xml_file}")
            if not json_file.exists():
                missing_files.append(f"Output: {json_file}")
        
        if missing_files:
            return False, f"Missing files: {', '.join(missing_files)}"
        return True, "ok"

    async def process_single_paper(
        self, paper_id: str, participant_url: str, updater: TaskUpdater
    ) -> PaperResult:
        """Process a single paper: load, send to participant, and evaluate.
        
        Args:
            paper_id: The paper ID to process
            participant_url: URL of the participant agent
            updater: Task updater for status updates
            
        Returns:
            PaperResult with evaluation results
        """
        logger.info(f"Processing Paper{paper_id}")
        
        try:
            # Load XML content
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Loading Paper{paper_id} XML..."),
            )
            xml_content = load_paper_xml(paper_id, self.input_dir)
            logger.info(f"Loaded XML for Paper{paper_id} ({len(xml_content)} chars)")
            
            # Build payload and send to participant
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Sending Paper{paper_id} to participant..."),
            )
            payload = build_participant_payload(xml_content, self.templates_dir)
            
            prediction = await self.messenger.talk_to_agent(
                message=payload,
                url=participant_url,
                new_conversation=True,  # isolate each paper
                timeout=self.timeout,
            )
            logger.info(f"Received response for Paper{paper_id} ({len(prediction)} chars)")
            
            # Load gold output
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Evaluating Paper{paper_id} response..."),
            )
            gold_output = load_gold_output(paper_id, self.output_dir)
            
            # Evaluate response
            evaluation_dict = evaluate_response(
                prediction=prediction,
                gold_output=gold_output,
                lang="en",
            )
            
            # Convert evaluation dict to PaperEvaluation model
            if "error" in evaluation_dict:
                evaluation = PaperEvaluation(
                    equation_match_percentage=0.0,
                    bertscore=BERTScore(precision=0.0, recall=0.0, f1=0.0),
                    error=evaluation_dict["error"],
                )
            else:
                bertscore_dict = evaluation_dict.get("bertscore", {})
                evaluation = PaperEvaluation(
                    equation_match_percentage=evaluation_dict.get(
                        "equation_match_percentage", 0.0
                    ),
                    bertscore=BERTScore(
                        precision=bertscore_dict.get("precision", 0.0),
                        recall=bertscore_dict.get("recall", 0.0),
                        f1=bertscore_dict.get("f1", 0.0),
                    ),
                )
            
            score = evaluation.equation_match_percentage
            logger.info(
                f"Paper{paper_id} evaluation complete: "
                f"equation_match={score:.2f}%, bertscore_f1={evaluation.bertscore.f1:.4f}"
            )
            
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Paper{paper_id} evaluation complete: "
                    f"equation_match={score:.2f}%, bertscore_f1={evaluation.bertscore.f1:.4f}"
                ),
            )
            
            return PaperResult(
                paper_id=paper_id,
                evaluation=evaluation,
                prediction=prediction,
                gold_output=gold_output,
            )
            
        except FileNotFoundError as e:
            error_msg = f"File not found: {str(e)}"
            logger.error(f"Paper{paper_id} error: {error_msg}")
            logger.debug(traceback.format_exc())
            return PaperResult(paper_id=paper_id, error=error_msg)
        except Exception as e:
            error_msg = f"Error processing paper: {str(e)}"
            logger.error(f"Paper{paper_id} error: {error_msg}")
            logger.debug(traceback.format_exc())
            return PaperResult(paper_id=paper_id, error=error_msg)

    def calculate_overall_metrics(
        self, paper_results: list[PaperResult]
    ) -> OverallResults:
        """Calculate overall metrics from paper results.
        
        Args:
            paper_results: List of paper evaluation results
            
        Returns:
            OverallResults with aggregated metrics
        """
        successful_results = [
            r for r in paper_results if r.evaluation is not None and r.error is None
        ]
        
        total_papers = len(self.paper_ids)
        
        if successful_results:
            mean_eq_match = sum(
                r.evaluation.equation_match_percentage for r in successful_results
            ) / len(successful_results)
            
            mean_bertscore_f1 = sum(
                r.evaluation.bertscore.f1 for r in successful_results
            ) / len(successful_results)
            
            overall_score = (mean_eq_match / 100.0 + mean_bertscore_f1) / 2.0
        else:
            mean_eq_match = 0.0
            mean_bertscore_f1 = 0.0
            overall_score = 0.0
        
        return OverallResults(
            overall_score=overall_score,
            mean_equation_match_percentage=mean_eq_match,
            mean_bertscore_f1=mean_bertscore_f1,
            total_papers=total_papers,
            successful_evaluations=len(successful_results),
            papers=paper_results,
        )

    async def process_papers(
        self, participant_url: str, updater: TaskUpdater
    ) -> list[PaperResult]:
        """Process all papers sequentially.
        
        Args:
            participant_url: URL of the participant agent
            updater: Task updater for status updates
            
        Returns:
            List of PaperResult objects
        """
        paper_results: list[PaperResult] = []
        total_papers = len(self.paper_ids)
        
        logger.info(f"Starting evaluation of {total_papers} papers")
        
        for idx, paper_id in enumerate(self.paper_ids):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Evaluating Paper{paper_id} ({idx+1}/{total_papers})..."
                ),
            )
            
            result = await self.process_single_paper(
                paper_id, participant_url, updater
            )
            paper_results.append(result)
        
        logger.info(f"Completed evaluation of {total_papers} papers")
        return paper_results

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Main entry point for the agent.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)
        """
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                logger.warning(f"Request validation failed: {msg}")
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            logger.error(f"Invalid request format: {e}")
            await updater.reject(
                new_agent_text_message(
                    "Invalid request. Expected a JSON EvalRequest like:\n"
                    + json.dumps(
                        {
                            "participants": {"participant": "http://localhost:9019"},
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + f"\n\nValidation error: {e}"
                )
            )
            return

        # Validate paper files exist
        ok, msg = self.validate_paper_files()
        if not ok:
            logger.error(f"Paper file validation failed: {msg}")
            await updater.reject(new_agent_text_message(msg))
            return

        participant_url = str(request.participants["participant"])

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Starting paper evaluation.\n{request.model_dump_json()}"
            ),
        )
        logger.info(f"Starting evaluation with request: {request.model_dump_json()}")

        # Process all papers
        paper_results = await self.process_papers(participant_url, updater)

        # Calculate overall metrics
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Paper processing finished. Calculating overall metrics..."),
        )
        logger.info("Paper processing finished. Calculating overall metrics.")

        overall_results = self.calculate_overall_metrics(paper_results)
        logger.info(f"Overall Results:\n{overall_results.model_dump_json()}")

        # Create result summary text
        result_text = (
            f"Overall score: {overall_results.overall_score:.4f}\n"
            f"Mean equation match: {overall_results.mean_equation_match_percentage:.2f}%\n"
            f"Mean BERTScore F1: {overall_results.mean_bertscore_f1:.4f}\n"
            f"Successful evaluations: {overall_results.successful_evaluations}/{overall_results.total_papers}"
        )

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=result_text)),
                Part(root=DataPart(data=overall_results.model_dump())),
            ],
            name="Result",
        )
