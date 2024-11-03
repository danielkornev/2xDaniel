"""
Agentic sampling loop that calls the Anthropic API and local implenmentation of anthropic-defined computer use tools.
"""
import asyncio
from pathlib import Path
import platform
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, cast
from uuid import uuid4

from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex, APIResponse
from anthropic.types import (
    ToolResultBlockParam,
)
from anthropic.types.beta import (
    BetaContentBlock,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock


from .tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult

from PIL import Image
from io import BytesIO
import gradio as gr
from typing import Dict

import openai
import os
import base64
from pydantic import BaseModel, Field, field_validator

openai.api_key = ""

BETA_FLAG = "computer-use-2024-10-22"
OUTPUT_DIR = "/tmp/outputs"

class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
}


# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilizing a Windows system with internet access.
* The current date is {datetime.today().strftime('%A, %B %d, %Y')}.
</SYSTEM_CAPABILITY>
"""

import base64
from PIL import Image
from io import BytesIO
import json

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def decode_base64_image_and_save(base64_str):
    # 移除base64字符串的前缀（如果存在）
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    # 解码base64字符串并将其转换为PIL图像
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    # 保存图像为screenshot.png
    import datetime
    image.save(f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    print("screenshot saved")
    return f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"

def decode_base64_image(base64_str):
    # 移除base64字符串的前缀（如果存在）
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    # 解码base64字符串并将其转换为PIL图像
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image


async def screenshot():
    """Take a screenshot of the current screen and return a ToolResult with the base64 encoded image."""
    import pyautogui
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"screenshot_{uuid4().hex}.png"

    # Take screenshot using pyautogui
    screenshot = pyautogui.screenshot()
    screenshot.save(str(path))

    if path.exists():
        # Return a ToolResult instance instead of a dictionary
        return ToolResult(base64_image=base64.b64encode(path.read_bytes()).decode())
        
    #raise ToolError(f"Failed to take screenshot: {path} does not exist.")


def screenshot_sync():
    """Take a screenshot of the current screen and return a ToolResult with the base64 encoded image."""
    import pyautogui
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"screenshot_{uuid4().hex}.png"

    # Take screenshot using pyautogui
    screenshot = pyautogui.screenshot()
    screenshot.save(str(path))

    if path.exists():
        # Return a ToolResult instance instead of a dictionary
        return ToolResult(base64_image=base64.b64encode(path.read_bytes()).decode())
    
    #raise ToolError(f"Failed to take screenshot: {path} does not exist.")


class BoundingBox(BaseModel):
    height: int = Field(description="height of bounding box")
    width: int = Field(description="width of bounding box")
    left: int = Field(description="x coordinate of the top left corner of the bounding box")
    top: int = Field(description="y coordinate of the top left corner of the bounding box")


class Evaluation(BaseModel):
    bounding_boxes: list[BoundingBox] = Field(description="List of bounding boxes where you should not click to satisfy policy")


def send_to_open_ai(base64_image, instruction):
    import requests
    import openai
    from openai import OpenAI

    response = openai.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        response_format=Evaluation,
    )

    # print(response.model_dump_json())

    return response



async def sampling_loop(
    *,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlock], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[[APIResponse[BetaMessage]], None],
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    tool_collection = ToolCollection(
        ComputerTool(),
        BashTool(),
        EditTool(),
    )
    system = (
        f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}"
    )

    while True:
        if only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(messages, only_n_most_recent_images)

        if provider == APIProvider.ANTHROPIC:
            client = Anthropic(api_key=api_key)
        elif provider == APIProvider.VERTEX:
            client = AnthropicVertex()
        elif provider == APIProvider.BEDROCK:
            client = AnthropicBedrock()

        # Call the API
        # we use raw_response to provide debug information to streamlit. Your
        # implementation may be able call the SDK directly with:
        # `response = client.messages.create(...)` instead.
        raw_response = client.beta.messages.with_raw_response.create(
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            system=system,
            tools=tool_collection.to_params(),
            betas=["computer-use-2024-10-22"],
        )

        api_response_callback(cast(APIResponse[BetaMessage], raw_response))

        response = raw_response.parse()

        messages.append(
            {
                "role": "assistant",
                "content": cast(list[BetaContentBlockParam], response.content),
            }
        )

        # call for GPT-4o-so with Vision
        

        tool_result_content: list[BetaToolResultBlockParam] = []
        for content_block in cast(list[BetaContentBlock], response.content):
            output_callback(content_block)

            # add blocker here
            # screenshot_base64 = (await screenshot()).base64_image

            # schema = send_to_open_ai(screenshot_base64).json()["choices"][0]["message"]["content"]
            # print(schema)

            #messages.append({"content": schema, "role": "assistant"})

            if content_block.type == "tool_use":
                result = await tool_collection.run(
                    name=content_block.name,
                    tool_input=cast(dict[str, Any], content_block.input),
                )
                tool_result_content.append(
                    _make_api_tool_result(result, content_block.id)
                )
                tool_output_callback(result, content_block.id)

        if not tool_result_content:
            return messages

        messages.append({"content": tool_result_content, "role": "user"})


def is_within_bounding_box(coordinate, box):
    x, y = coordinate
    left, top, width, height = box['left'], box['top'], box['width'], box['height']
    return left <= x <= left + width and top <= y <= top + height


def sampling_loop_sync(
    *,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlock], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[[APIResponse[BetaMessage]], None],
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
):
    """
    Synchronous agentic sampling loop for the assistant/tool interaction of computer use.
    """
    tool_collection = ToolCollection(
        ComputerTool(),
        BashTool(),
        EditTool(),
    )
    system = (
        f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}"
    )

    while True:
        if only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(messages, only_n_most_recent_images)

        # Instantiate the appropriate API client based on the provider
        if provider == APIProvider.ANTHROPIC:
            client = Anthropic(api_key=api_key)
        elif provider == APIProvider.VERTEX:
            client = AnthropicVertex()
        elif provider == APIProvider.BEDROCK:
            client = AnthropicBedrock()

        # Call the API synchronously
        raw_response = client.beta.messages.with_raw_response.create(
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            system=system,
            tools=tool_collection.to_params(),
            betas=["computer-use-2024-10-22"],
        )

        api_response_callback(cast(APIResponse[BetaMessage], raw_response))

        response = raw_response.parse()

        # Append new assistant message
        new_message = {
            "role": "assistant",
            "content": cast(list[BetaContentBlockParam], response.content),
        }
        messages.append(new_message)

        tool_result_content: list[BetaToolResultBlockParam] = []
        for content_block in cast(list[BetaContentBlock], response.content):
            output_callback(content_block)

            # add blocker here
            screenshot_base64 = (screenshot_sync()).base64_image
            instruction = "Don't allow any financial transactions"

            bounding_boxes = json.loads(send_to_open_ai(screenshot_base64, instruction).model_dump_json())["choices"][0]["message"]["parsed"]
            print(bounding_boxes)
            

            # messages.append({"content": str(schema), "role": "assistant"})

            if content_block.type == "tool_use":
                print(str(content_block))
                action = content_block.input["action"]
                if action == "mouse_move":
                    coordinate = content_block.input["coordinate"]
                    within_any_box = any(is_within_bounding_box(coordinate, box) for box in bounding_boxes['bounding_boxes'])

                    # # Print the result
                    # if within_any_box:
                    #     print("no")
                    #     # hacky
                    #     content_block = BetaToolUseBlock(id=content_block.id, input={action = "mouse_move", coordinate = [0,0], type})
                    #     content_block.input["coordinate"] = [0,0]
                    # else:
                    #     continue

                # Run the asynchronous tool execution in a synchronous context
                result = asyncio.run(tool_collection.run(
                    name=content_block.name,
                    tool_input=cast(dict[str, Any], content_block.input),
                ))
                tool_result_content.append(
                    _make_api_tool_result(result, content_block.id)
                )
                tool_output_callback(result, content_block.id)

            # Craft messages based on the content_block
            res = []
            for msg in messages:
                try:
                    if isinstance(msg["content"][0], TextBlock):
                        res.append((msg["content"][0].text, None))  # User message
                    elif isinstance(msg["content"][0], BetaTextBlock):
                        res.append((None, msg["content"][0].text))  # Bot message
                    elif isinstance(msg["content"][0], BetaToolUseBlock):
                        res.append((None, f"Tool Use: {msg['content'][0].name}\nInput: {msg['content'][0].input}"))  # Bot message
                    elif isinstance(msg["content"][0], Dict) and msg["content"][0]["content"][-1]["type"] == "image":
                        # image_path = decode_base64_image_and_save(msg["content"][0]["content"][-1]["source"]["data"])
                        # res.append((None, gr.Image(image_path)))  # Bot message
                        res.append((None, f'<img src="data:image/png;base64,{msg["content"][0]["content"][-1]["source"]["data"]}">'))  # Bot message
                    else:
                        print(msg["content"][0])
                except Exception as e:
                    print("error", e)
                    pass
            
            # Yield crafted messages
            for user_msg, bot_msg in res:
                yield [user_msg, bot_msg]

        if not tool_result_content:
            return messages

        messages.append({"content": tool_result_content, "role": "user"})


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 10,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[ToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
