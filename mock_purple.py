import argparse
import os
import asyncio
import uvicorn
import argparse
import os

import uvicorn
from dotenv import load_dotenv
from litellm import completion
from loguru import logger

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message


load_dotenv()

DEFAULT_MODEL = os.getenv("PURPLE_MODEL", "openai/gpt-4.1-mini")

SYSTEM_PROMPT = (
    "You are a helpful extraction assistant. Follow the policies and templates provided "
    "in each request, and respond with valid JSON only."
)


def prepare_agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="task_fulfillment",
        name="Task Fulfillment",
        description="Generates structured JSON responses for dairy-paper extraction tasks.",
        tags=["participant", "benchmark"],
        examples=[],
    )

    return AgentCard(
        name="Mock Purple Agent",
        description="Local purple agent that proxies prompts to an LLM following tau2 template.",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


class MockPurpleExecutor(AgentExecutor):
    def __init__(self, model: str):
        self.model = model
        self.ctx_id_to_messages: dict[str, list[dict]] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        if user_input is None:
            logger.warning("Received empty user input; replying with error message.")
            await event_queue.enqueue_event(
                new_agent_text_message(
                    "{\"error\": \"No input provided by green agent.\"}",
                    context_id=context.context_id,
                )
            )
            return

        logger.info(f"Received input: {user_input[:200]}...")

        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

        messages = self.ctx_id_to_messages[context.context_id]
        messages.append({"role": "user", "content": user_input})

        try:
            response = completion(
                messages=messages,
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            assistant_content = response.choices[0].message.content or "{}"
            logger.info(f"LLM response: {assistant_content[:200]}...")
        except Exception as exc:
            logger.error(f"LLM error: {exc}")
            assistant_content = (
                '{"error": "LLM call failed", "details": "' + str(exc) + '"}'
            )

        messages.append({"role": "assistant", "content": assistant_content})

        await event_queue.enqueue_event(
            new_agent_text_message(assistant_content, context_id=context.context_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Run the mock purple agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL for the agent card")
    parser.add_argument(
        "--agent-llm",
        type=str,
        default=os.getenv("PURPLE_MODEL", DEFAULT_MODEL),
        help="LLM identifier understood by litellm",
    )
    args = parser.parse_args()

    logger.info("Starting mock purple agent...")
    card = prepare_agent_card(args.card_url or f"http://{args.host}:{args.port}/")

    request_handler = DefaultRequestHandler(
        agent_executor=MockPurpleExecutor(model=args.agent_llm),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(
        app.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()
