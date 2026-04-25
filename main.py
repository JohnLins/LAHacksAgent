import os
import time
import traceback
from datetime import datetime
from uuid import uuid4

import requests
from uagents import Agent, Context, Model, Protocol
from uagents.setup import fund_agent_if_low
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)

try:
    # Works when running from repo root (e.g. `python -m agent.main`)
    from agent.extractlabor import extract_human_tasks_from_prompt
except Exception:
    # Works when Railway service root is `agent/` (e.g. `python main.py`)
    from extractlabor import extract_human_tasks_from_prompt


AGENT_NAME = os.getenv("AGENT_NAME", "HumanAgent")
AGENT_SEED = os.getenv("AGENT_SEED", "lahacks-fetch-agent-dev-seed-change-me")
AGENT_PORT = int(os.getenv("AGENT_PORT", os.getenv("PORT", "8001")))
AGENT_ENDPOINT = os.getenv("AGENT_ENDPOINT")

# Defaults to your deployed marketplace backend.
# You can override with MARKETPLACE_URL in Railway variables.
MARKETPLACE_URL = os.getenv(
    "MARKETPLACE_URL",
    "https://lahacksbackend-production.up.railway.app/api/tasks/",
)


agent = Agent(
    name=AGENT_NAME,
    seed=AGENT_SEED,
    port=AGENT_PORT,
    endpoint=AGENT_ENDPOINT,
    network=os.getenv("AGENT_NETWORK", "testnet"),
)
fund_agent_if_low(agent.wallet.address())

chat_proto = Protocol(spec=chat_protocol_spec)

@agent.on_event("startup")
async def _startup(ctx: Context):
    """
    Optional: auto-register this uAgent with Agentverse ACP from Railway.

    If you set an Agentverse API key, the agent will attempt registration on boot,
    so you don't need to run any local "registration script".
    """
    api_key = (
        os.getenv("AGENTVERSE_API_KEY")
        or os.getenv("ILABS_AGENTVERSE_API_KEY")
        or os.getenv("AGENTVERSE_KEY")
    )
    if not api_key:
        ctx.logger.info("Agentverse API key not set; skipping ACP registration.")
        return

    # Agentverse needs a public, reachable endpoint for ACP verification.
    if not AGENT_ENDPOINT:
        ctx.logger.warning("AGENT_ENDPOINT not set; skipping ACP registration.")
        return

    try:
        from uagents_core.utils.registration import register_chat_agent  # type: ignore
    except Exception as exc:
        ctx.logger.warning(
            f"ACP registration import failed; skipping registration: {exc}"
        )
        return

    try:
        result = register_chat_agent(
            name=AGENT_NAME,
            api_key=api_key,
            seed=AGENT_SEED,
            endpoint=AGENT_ENDPOINT,
        )
        ctx.logger.info(f"ACP registration result: {result}")
    except Exception as exc:
        ctx.logger.warning(f"ACP registration failed: {exc}")


class HealthResponse(Model):
    status: str
    agent: str
    address: str


@agent.on_rest_get("/", HealthResponse)
async def health(_: Context) -> HealthResponse:
    return HealthResponse(status="ok", agent=AGENT_NAME, address=agent.address)


def _create_text_chat(text: str) -> ChatMessage:
    return ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=[TextContent(type="text", text=text)],
    )

def _safe_preview(text: str | None, limit: int = 500) -> str:
    if not text:
        return ""
    t = str(text).replace("\n", "\\n")
    if len(t) <= limit:
        return t
    return t[:limit] + f"...(+{len(t) - limit} chars)"


def _safe_env_summary() -> dict[str, str]:
    # Don't log secrets; only log presence/shape.
    return {
        "agent_name": AGENT_NAME,
        "agent_port": str(AGENT_PORT),
        "agent_endpoint_set": str(bool(AGENT_ENDPOINT)),
        "marketplace_url": MARKETPLACE_URL,
        "agentverse_api_key_set": str(
            bool(
                os.getenv("AGENTVERSE_API_KEY")
                or os.getenv("ILABS_AGENTVERSE_API_KEY")
                or os.getenv("AGENTVERSE_KEY")
            )
        ),
        "coralflavor_api_key_set": str(
            bool(os.getenv("CORALFLAVOR_API_KEY") or os.getenv("CORAL_API_KEY"))
        ),
    }


@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    req_id = uuid4().hex[:12]
    t0 = time.time()
    ctx.logger.info(
        f"[{req_id}] chat_message received sender={sender} msg_id={msg.msg_id}"
    )
    ctx.logger.info(f"[{req_id}] env={_safe_env_summary()}")
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.utcnow(),
            acknowledged_msg_id=msg.msg_id,
        ),
    )

    ctx.logger.info(f"[{req_id}] content_items={len(msg.content)}")
    for idx, item in enumerate(msg.content):
        if isinstance(item, StartSessionContent):
            ctx.logger.info(f"[{req_id}] item[{idx}] StartSessionContent")
            continue

        if isinstance(item, EndSessionContent):
            ctx.logger.info(f"[{req_id}] item[{idx}] EndSessionContent")
            continue

        if not isinstance(item, TextContent):
            ctx.logger.info(
                f"[{req_id}] item[{idx}] unexpected_content_type={type(item)}"
            )
            continue

        prompt = item.text
        ctx.logger.info(
            f"[{req_id}] item[{idx}] text_len={len(prompt or '')} preview={_safe_preview(prompt, 300)}"
        )

        try:
            t_extract0 = time.time()
            tasks = extract_human_tasks_from_prompt(prompt)
            ctx.logger.info(
                f"[{req_id}] extractlabor ok tasks={len(tasks)} elapsed_ms={int((time.time()-t_extract0)*1000)}"
            )
            if not tasks:
                reply = "No human tasks found in the prompt."
            else:
                posted = 0
                failures: list[str] = []
                for ti, task in enumerate(tasks):
                    description = task.get("task")
                    compensation = float(task.get("compensation", 0))
                    ctx.logger.info(
                        f"[{req_id}] post[{ti}] -> marketplace desc_preview={_safe_preview(str(description), 120)} comp={compensation}"
                    )
                    t_post0 = time.time()
                    try:
                        response = requests.post(
                            MARKETPLACE_URL,
                            json={
                                "description": description,
                                "compensation": compensation,
                            },
                            timeout=15,
                        )
                    except Exception as post_exc:
                        failures.append(str(description))
                        ctx.logger.error(
                            f"[{req_id}] post[{ti}] exception={post_exc}\n{traceback.format_exc()}"
                        )
                        continue
                    ctx.logger.info(
                        f"[{req_id}] post[{ti}] status={response.status_code} elapsed_ms={int((time.time()-t_post0)*1000)}"
                    )
                    if response.ok:
                        posted += 1
                        ctx.logger.info(f"[{req_id}] post[{ti}] ok")
                    else:
                        failures.append(str(description))
                        ctx.logger.error(
                            f"[{req_id}] post[{ti}] failed body_preview={_safe_preview(response.text, 500)}"
                        )

                reply = f"Posted {posted} of {len(tasks)} extracted human tasks."
                if failures:
                    reply += f" Failed: {', '.join(failures)}"
        except Exception as exc:
            ctx.logger.error(
                f"[{req_id}] handler exception={exc}\n{traceback.format_exc()}"
            )
            reply = (
                "Internal error while extracting or posting tasks. "
                f"(ref: {req_id})"
            )

        ctx.logger.info(
            f"[{req_id}] sending_reply len={len(reply)} elapsed_ms={int((time.time()-t0)*1000)}"
        )
        await ctx.send(sender, _create_text_chat(reply))


@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(
        f"Received acknowledgement from {sender} for message {msg.acknowledged_msg_id}"
    )


agent.include(chat_proto, publish_manifest=True)


if __name__ == "__main__":
    agent.run()

