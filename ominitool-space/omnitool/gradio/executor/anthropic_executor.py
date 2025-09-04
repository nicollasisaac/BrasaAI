# executor/anthropic_executor.py
import asyncio
from typing import Any, Dict, Iterable, Tuple, Callable, List, Optional

from anthropic.types import TextBlock
from anthropic.types.beta import (
    BetaContentBlock,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)

from tools import ComputerTool, ToolCollection, ToolResult


class AnthropicExecutor:
    """
    Executor para o fluxo Anthropics (Computer Use).
    Observação: o caminho OCI (omniparser + gpt-4.1-oci) NÃO usa este executor em loop.py.
    Mantemos aqui para compatibilidade com os outros modelos.
    """

    def __init__(
        self,
        output_callback: Callable[[BetaContentBlockParam], None],
        tool_output_callback: Callable[[Any, str], None],
    ):
        # Por padrão, usa o backend LOCAL (pyautogui) do ComputerTool.
        # Se WINDOWS_HOST_URL for definido ou COMPUTER_BACKEND=http, o ComputerTool
        # fará fallback automaticamente para o backend HTTP.
        self.tool_collection = ToolCollection(ComputerTool())
        self.output_callback = output_callback
        self.tool_output_callback = tool_output_callback

    def __call__(
        self,
        response: BetaMessage,
        messages: List[BetaMessageParam],
    ) -> Iterable[Tuple[List[Any] | None, List[BetaToolResultBlockParam]]]:
        """
        Itera pelos content blocks da resposta do modelo Anthropic,
        dispara callbacks para renderização e executa ferramentas (tool_use).

        Yields:
            (message_stub, tool_result_content)
            - message_stub é mantido como [None, None] para compat com o app,
              pois a renderização real acontece via output_callback.
            - tool_result_content acumula os BetaToolResultBlockParam para o follow-up.
        """
        # Garante que a resposta do assistente vai para o histórico (se necessário)
        new_message = {
            "role": "assistant",
            "content": list(response.content),  # type: ignore[arg-type]
        }
        if new_message not in messages:
            messages.append(new_message)
        else:
            print("new_message already in messages, duplicates detected.")

        tool_result_content: List[BetaToolResultBlockParam] = []

        # Percorre blocos da resposta
        for content_block in list(response.content):  # type: ignore[call-arg]
            # Renderiza no chat (o output_callback do app cuida do HTML/formatos)
            try:
                self.output_callback(content_block)  # sender="bot" é tratado no callback do app
            except TypeError:
                # Compat com versões do callback que aceitam 'sender'
                self.output_callback(content_block, sender="bot")  # type: ignore[misc]

            # Execução de ferramentas (tool_use)
            if getattr(content_block, "type", None) == "tool_use":
                # Executa de forma síncrona (o run interno é async)
                result: ToolResult = asyncio.run(
                    self.tool_collection.run(
                        name=content_block.name,  # type: ignore[attr-defined]
                        tool_input=dict(content_block.input),  # type: ignore[attr-defined]
                    )
                )

                # Mostra saída/erro/imagem da tool no chat
                try:
                    self.output_callback(result)  # pode aceitar sender opcional
                except TypeError:
                    self.output_callback(result, sender="bot")  # type: ignore[misc]

                # Dispara callback externo (p/ logging/tracing)
                try:
                    self.tool_output_callback(result, content_block.id)  # type: ignore[attr-defined]
                except Exception as e:
                    print(f"tool_output_callback error: {e}")

                # Empacota ToolResult -> Anthropic ToolResultBlockParam
                tool_result_content.append(
                    _make_api_tool_result(result, content_block.id)  # type: ignore[arg-type]
                )

            # “tick” para o loop do Gradio/Streamlit
            display_messages = _message_display_callback(messages)
            for _user_msg, _bot_msg in display_messages:
                yield [None, None], tool_result_content

        # Se não houve ferramentas, devolve apenas mensagens (compat com chamador)
        if not tool_result_content:
            return messages  # type: ignore[return-value]

        # Caso contrário, devolve results para follow-up
        return tool_result_content  # type: ignore[return-value]


def _message_display_callback(messages: List[BetaMessageParam]) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Constrói uma visão simplificada (user_msg, bot_msg) para manter a compat
    com o loop do app, enquanto a renderização real já ocorreu via output_callback.
    """
    display_messages: List[Tuple[Optional[str], Optional[str]]] = []
    for msg in messages:
        try:
            content = msg.get("content", [])
            if not content:
                continue
            first = content[0]

            # Mensagem do usuário (TextBlock "padrão")
            if isinstance(first, TextBlock):
                display_messages.append((first.text, None))
            # Mensagem do bot (texto Anthropic beta)
            elif isinstance(first, BetaTextBlock):
                display_messages.append((None, first.text))
            # tool_use – só um resumo textual
            elif getattr(first, "type", None) == "tool_use":
                name = getattr(first, "name", "tool")
                tool_in = getattr(first, "input", {})
                display_messages.append((None, f"Tool Use: {name}\nInput: {tool_in}"))
            # imagem antiga (dict com base64)
            elif isinstance(first, dict) and first.get("content"):
                last = first["content"][-1]
                if isinstance(last, dict) and last.get("type") == "image":
                    try:
                        b64 = last["source"]["data"]
                        display_messages.append((None, f'<img src="data:image/png;base64,{b64}">'))
                    except Exception:
                        pass
        except Exception as e:
            print("display callback error:", e)
            continue
    return display_messages


def _make_api_tool_result(result: ToolResult, tool_use_id: str) -> BetaToolResultBlockParam:
    """
    Converte um ToolResult do agente para o formato Anthropic (ToolResultBlockParam).
    """
    tool_result_content: List[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False

    if getattr(result, "error", None):
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)  # type: ignore[assignment]
    else:
        if getattr(result, "output", None):
            tool_result_content.append(  # type: ignore[union-attr]
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),  # type: ignore[arg-type]
                }
            )
        if getattr(result, "base64_image", None):
            tool_result_content.append(  # type: ignore[union-attr]
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,  # type: ignore[arg-type]
                    },
                }
            )

    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str) -> str:
    if getattr(result, "system", None):
        return f"<system>{result.system}</system>\n{result_text}"  # type: ignore[operator]
    return result_text
