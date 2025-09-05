"""
Agentic sampling loop that supports:
- Anthropic Computer Use (original)
- VLM agents (original)
- OmniParser + GPT-4.1 (OCI)

Quando o modelo é "omniparser + gpt-4.1-oci" (ou "-orchestrated"):
- O agente (VLMAgent / VLMOrchestratedAgent) faz o screenshot local e chama o OmniParser internamente.
- O loop não chama OmniParserClient nem salva screenshots — só orquestra chamadas.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any, Iterable, List

# ===== Tipos Anthropic (para compatibilidade com o app/gradio) =====
from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import (
    BetaContentBlock,
    BetaMessage,
    BetaMessageParam,
)

# ===== Infra original =====
from tools import ToolResult
from agent.llm_utils.omniparserclient import OmniParserClient
from agent.anthropic_agent import AnthropicActor
from agent.vlm_agent import VLMAgent
from agent.vlm_agent_with_orchestrator import VLMOrchestratedAgent
from executor.anthropic_executor import AnthropicExecutor


# ===================== Providers =====================
class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    OPENAI = "openai"
    OCI = "oci"


# (mantido para compat, caso o app use)
PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
    APIProvider.OPENAI: "gpt-4o",
    # OCI usa o rótulo "omniparser + gpt-4.1-oci" no seletor do app
}


def sampling_loop_sync(
    *,
    model: str,
    provider: APIProvider | None,
    messages: List[BetaMessageParam],
    output_callback: Callable[[BetaContentBlock], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[[APIResponse[BetaMessage]], None],
    api_key: str,
    only_n_most_recent_images: int | None = 2,
    max_tokens: int = 4096,
    omniparser_url: str,
    save_folder: str = "./uploads",
) -> Iterable[list[Any] | None]:
    """
    Synchronous agentic sampling loop for the assistant/tool interaction.

    Fluxos:
    - "claude-3-5-sonnet-20241022": Anthropic Computer Use (ator Anthropic + executor)
    - VLMs "omniparser + gpt-4o"/"o1"/"o3-mini"/"R1"/"qwen2.5vl": OmniParserClient fornece parsed_screen; VLM decide ação
    - VLMs "-orchestrated": idem acima, mas com orquestrador
    - "omniparser + gpt-4.1-oci" (e "-orchestrated"): o agente captura a tela local + chama OmniParser direto; aqui passamos parsed_screen=None
    """
    print("in sampling_loop_sync, model:", model)

    # ====== Seleção do ator conforme o modelo ======
    if model == "claude-3-5-sonnet-20241022":
        actor = AnthropicActor(
            model=model,
            provider=provider,
            api_key=api_key,
            api_response_callback=api_response_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images,
        )
    elif model in {
        "omniparser + gpt-4o",
        "omniparser + o1",
        "omniparser + o3-mini",
        "omniparser + R1",
        "omniparser + qwen2.5vl",
        "omniparser + gpt-4.1-oci",  # OCI simples (sem orquestrador)
    }:
        actor = VLMAgent(
            model=model,
            provider=provider,
            api_key=api_key,
            api_response_callback=api_response_callback,
            output_callback=output_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images,
        )
    elif model in {
        "omniparser + gpt-4o-orchestrated",
        "omniparser + o1-orchestrated",
        "omniparser + o3-mini-orchestrated",
        "omniparser + R1-orchestrated",
        "omniparser + qwen2.5vl-orchestrated",
        "omniparser + gpt-4.1-oci-orchestrated",  # OCI com orquestrador
    }:
        actor = VLMOrchestratedAgent(
            model=model,
            provider=provider,
            api_key=api_key,
            api_response_callback=api_response_callback,
            output_callback=output_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images,
            save_folder=save_folder,
        )
    else:
        raise ValueError(f"Model {model} not supported")

    # Executor Anthropic (centraliza execução das tool_use que o ator emitir)
    executor = AnthropicExecutor(
        output_callback=output_callback,
        tool_output_callback=tool_output_callback,
    )

    print(f"Model Inited: {model}, Provider: {provider}")
    print(f"Start the message loop. User messages: {messages}")

    # ====== Fluxo Anthropic Computer Use ======
    if model == "claude-3-5-sonnet-20241022":
        # Para Anthropic, capturamos a tela via OmniParserClient a cada passo
        omniparser_client = OmniParserClient(url=omniparser_url)

        while True:
            parsed_screen = omniparser_client()  # dict com som_image_base64/parsed_content_list/screen_info/etc
            screen_info_block = TextBlock(
                text=(
                    "Below is the structured accessibility information of the current UI screen, "
                    "which includes text and icons you can operate on. Take this information into account "
                    "when you are making the prediction for the next action. Note you will still need "
                    "to take a screenshot to get the image:\n"
                    + parsed_screen["screen_info"]
                ),
                type="text",
            )
            screen_info_dict = {"role": "user", "content": [screen_info_block]}
            messages.append(screen_info_dict)

            tools_use_needed = actor(messages=messages)

            for message, tool_result_content in executor(tools_use_needed, messages):
                # Gradio espera yields de mensagens (tuplas) para atualizar UI
                yield message

            # Se não tiver tool_result_content, encerramos
            if not tool_result_content:
                return

            messages.append({"content": tool_result_content, "role": "user"})

    # ====== Fluxos VLM (inclui OCI) ======
    elif model in {
        "omniparser + gpt-4o",
        "omniparser + o1",
        "omniparser + o3-mini",
        "omniparser + R1",
        "omniparser + qwen2.5vl",
        "omniparser + gpt-4.1-oci",
        "omniparser + gpt-4o-orchestrated",
        "omniparser + o1-orchestrated",
        "omniparser + o3-mini-orchestrated",
        "omniparser + R1-orchestrated",
        "omniparser + qwen2.5vl-orchestrated",
        "omniparser + gpt-4.1-oci-orchestrated",
    }:
        # Para VLMs não-OCI, usamos o OmniParserClient (endpoint). Para OCI, o agente fará screenshot+Omni internamente.
        omniparser_client = OmniParserClient(url=omniparser_url)

        while True:
            if model in {"omniparser + gpt-4.1-oci", "omniparser + gpt-4.1-oci-orchestrated"}:
                parsed_screen = None
            else:
                parsed_screen = omniparser_client()

            tools_use_needed, vlm_response_json = actor(
                messages=messages,
                parsed_screen=parsed_screen,
            )

            for message, tool_result_content in executor(tools_use_needed, messages):
                yield message

            if not tool_result_content:
                return
