import asyncio
import json
import httpx
from uuid import uuid4

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart


GREEN_URL = "http://127.0.0.1:9009/"

async def main():
    # This is the payload your Agent.run() expects as message text
    eval_request = {
        "participants": {"participant": "http://127.0.0.1:9019/"}
    }

    async with httpx.AsyncClient(timeout=500) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=GREEN_URL)
        agent_card = await resolver.get_agent_card()

        client = ClientFactory(ClientConfig(httpx_client=httpx_client, streaming=False)).create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=json.dumps(eval_request)))],
            message_id=uuid4().hex,
            context_id=None,
        )

        events = [event async for event in client.send_message(msg)]

    # Print what came back (task + artifacts usually show up as (task, update))
    for event in events:
        print(event)

if __name__ == "__main__":
    asyncio.run(main())
