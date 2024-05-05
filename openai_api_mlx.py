import os
import gc
import time
import base64
import re
import argparse

from contextlib import asynccontextmanager
from typing import List, Literal, Union, Tuple, Optional
import uvicorn
import mlx
import mlx.core as mx
import mlx_lm
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from llava.generate import load_model, prepare_inputs, generate_text
from llava.llava import LlavaModel
from autogen.agentchat.contrib.img_utils import get_image_data, llava_formatter
from autogen.code_utils import content_str
from PIL import Image
from io import BytesIO

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    An asynchronous context manager for managing the lifecycle of the FastAPI app.
    It ensures that GPU memory is cleared after the app's lifecycle ends, which is essential for efficient resource management in GPU environments.
    """
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    """
    A Pydantic model representing a model card, which provides metadata about a machine learning model.
    It includes fields like model ID, owner, and creation time.
    """

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ImageUrl(BaseModel):
    url: str


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrlContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentItem = Union[TextContent, ImageUrlContent]


class ChatMessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[ContentItem]]
    name: Optional[str] = None


class ChatMessageResponse(BaseModel):
    role: Literal["assistant"]
    content: str = None
    name: Optional[str] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessageInput]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    # Additional parameters
    repetition_penalty: Optional[float] = 1.0

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessageResponse


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[
        Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
    ]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """
    An endpoint to list available models. It returns a list of model cards.
    This is useful for clients to query and understand what models are available for use.
    """
    model_card_llava = ModelCard(id="llava-1.5-7b-hf")
    model_card_cog = ModelCard(
        id="cogvlm-chat-17b"
    )  # can be replaced by your model id like cogagent-chat-18b
    return ModelList(data=[model_card_llava, model_card_cog])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    logger.debug(f"==== model ====\n{request.model}")

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    if "llava" in request.model:
        gen_params = dict(
            messages=request.messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 1024,
            echo=False,
            stream=request.stream,
        )
        response = generate_llava(model, gen_params)
    elif "cog" in request.model:
        gen_params = dict(
            messages=request.messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 1024,
            echo=False,
            stream=request.stream,
        )

        if request.stream:
            generate = predict(request.model, gen_params)
            return EventSourceResponse(generate, media_type="text/event-stream")
        response = generate_cogvlm(model, tokenizer, gen_params)

    usage = UsageInfo()

    message = ChatMessageResponse(
        role="assistant",
        content=response["text"],
    )
    logger.debug(f"==== message ====\n{message}")
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
    )
    task_usage = UsageInfo.model_validate(response["usage"])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
    return ChatCompletionResponse(
        model=request.model,
        choices=[choice_data],
        object="chat.completion",
        usage=usage,
    )


async def predict(model_id: str, params: dict):
    """
    Handle streaming predictions. It continuously generates responses for a given input stream.
    This is particularly useful for real-time, continuous interactions with the model.
    """

    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(role="assistant"), finish_reason=None
    )
    chunk = ChatCompletionResponse(
        model=model_id, choices=[choice_data], object="chat.completion.chunk"
    )
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    previous_text = ""
    for new_response in generate_stream_cogvlm(model, tokenizer, params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(previous_text) :]
        previous_text = decoded_unicode
        delta = DeltaMessage(
            content=delta_text,
            role="assistant",
        )
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=delta,
        )
        chunk = ChatCompletionResponse(
            model=model_id, choices=[choice_data], object="chat.completion.chunk"
        )
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
    )
    chunk = ChatCompletionResponse(
        model=model_id, choices=[choice_data], object="chat.completion.chunk"
    )
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

def generate_llava(
    model: LlavaModel, params: dict
):
    """
    Generates a response using the Llava model. It processes the chat history and image data, if any,
    and then invokes the model to generate a response.
    """

    messages = params["messages"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))

    # The formats for LLaVA and CogVLM are different. So, we manually handle them here.
    prompt, image_list = process_llava_messages(messages)

    logger.debug(f"==== request ====\n{prompt}")
    logger.debug(f"==== image ====\n{image_list[-1]}")

    if len(image_list) > 0:
        input_ids, pixel_values = prepare_inputs(processor, image_list[-1], prompt)
        input_echo_len = len(input_ids[0])

        total_len = 0
        generated_text = ""

        generated_text = generate_text(
            input_ids, pixel_values, model, processor, max_new_tokens, temperature
        )

        response = {
            "text": generated_text,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
            },
        }
    else:
        generated_text = "An image input is required!"
        response = {
            "text": generated_text,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    return response

def generate_cogvlm(
    model, tokenizer, params: dict
):
    """
    Generates a response using the CogVLM model. It processes the chat history and image data, if any,
    and then invokes the model to generate a response.
    """

    for response in generate_stream_cogvlm(model, tokenizer, params):
        pass
    return response

def process_llava_messages(
    messages: List[ChatMessageInput],
) -> Tuple[Optional[str], Optional[List[Image.Image]]]:
    """
    Process llava messages to extract prompt, identify the last user query,
    and convert base64 encoded image URLs to PIL images.

    Args:
        messages(List[ChatMessageInput]): List of ChatMessageInput objects.
    return: A tuple of three elements:
             - The formatted prompt for llava as a string.
             - List of PIL Image objects extracted from the messages.
    """
    formatted_prompt = ""
    image_list = []
    # Increment the image count and replace the tag in the prompt
    new_token = ""

    for i, message in enumerate(messages):
        role = message.role
        content = message.content

        if isinstance(content, list):  # text
            text_content = " ".join(
                item.text for item in content if isinstance(item, TextContent)
            )
        else:
            text_content = content

        logger.debug(f"==== text content ====\n{text_content}")
        if isinstance(content, list):  # image
            for item in content:
                if isinstance(item, ImageUrlContent):
                    image_url = item.image_url.url
                    if re.match("data:image/.+;base64,", image_url):
                        base64_encoded_image = re.sub(
                            "data:image/.+;base64,", "", image_url, count=1
                        )
                        image_data = base64.b64decode(base64_encoded_image)
                        image = Image.open(BytesIO(image_data)).convert("RGB")
                        image_list.append(image)
                        new_token = "<image>\n"


        formatted_prompt += f"{role.upper()}: {new_token}{text_content}</s>"
        new_token = ""

    formatted_prompt += "ASSISTANT: "

    return formatted_prompt, image_list

def process_history_and_images(
    messages: List[ChatMessageInput],
) -> Tuple[Optional[str], Optional[List[Tuple[str, str]]], Optional[List[Image.Image]]]:
    """
    Process history messages to extract text, identify the last user query,
    and convert base64 encoded image URLs to PIL images.

    Args:
        messages(List[ChatMessageInput]): List of ChatMessageInput objects.
    return: A tuple of three elements:
             - The last user query as a string.
             - Text history formatted as a list of tuples for the model.
             - List of PIL Image objects extracted from the messages.
    """
    formatted_history = []
    image_list = []
    last_user_query = ""

    for i, message in enumerate(messages):
        role = message.role
        content = message.content

        if isinstance(content, list):  # text
            text_content = " ".join(
                item.text for item in content if isinstance(item, TextContent)
            )
        else:
            text_content = content

        if isinstance(content, list):  # image
            for item in content:
                if isinstance(item, ImageUrlContent):
                    image_url = item.image_url.url
                    if re.match("data:image/.+;base64,", image_url):
                        base64_encoded_image = re.sub(
                            "data:image/.+;base64,", "", image_url, count=1
                        )
                        image_data = base64.b64decode(base64_encoded_image)
                        image = Image.open(BytesIO(image_data)).convert("RGB")
                        image_list.append(image)

        if role == "user":
            if i == len(messages) - 1:  # 最后一条用户消息
                last_user_query = text_content
            else:
                formatted_history.append((text_content, ""))
        elif role == "assistant":
            if formatted_history:
                if formatted_history[-1][1] != "":
                    assert (
                        False
                    ), f"the last query is answered. answer again. {formatted_history[-1][0]}, {formatted_history[-1][1]}, {text_content}"
                formatted_history[-1] = (formatted_history[-1][0], text_content)
            else:
                assert False, f"assistant reply before user"
        else:
            assert False, f"unrecognized role: {role}"

    return last_user_query, formatted_history, image_list

def generate_stream_cogvlm(
    model, tokenizer, params: dict
):
    """
    Generates a stream of responses using the CogVLM model in inference mode.
    It's optimized to handle continuous input-output interactions with the model in a streaming manner.
    """
    messages = params["messages"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    query, history, image_list = process_history_and_images(messages)

    logger.debug(f"==== request ====\n{query}")

    generated_text = "CogVLM model is not supported with MLX!"
    ret = {
        "text": generated_text,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    yield ret


gc.collect()
mx.metal.clear_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="OpenAI API Local Server",
        description="",
    )
    parser.add_argument('--model_path', default="THUDM/cogvlm-chat-hf")
    parser.add_argument('-q', '--quant', default=False, action='store_true')
    parser.add_argument('--tokenizer_path', default="lmsys/vicuna-7b-v1.5")
    args = parser.parse_args()

    dtype = mx.float16

    print(
        "========Use dtype as:{} with MLX========\n\n".format(
            dtype
        )
    )

    processor, model = load_model(args.model_path)

    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)
