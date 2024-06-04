import os
import gc
import time
import base64
import re
import argparse

from contextlib import asynccontextmanager
from typing import List, Literal, Union, Tuple, Optional
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TextIteratorStreamer,
)
from PIL import Image
from io import BytesIO

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    An asynchronous context manager for managing the lifecycle of the FastAPI app.
    It ensures that GPU memory is cleared after the app's lifecycle ends, which is essential for efficient resource management in GPU environments.
    """
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


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
    cogvlm_model_card = ModelCard(
        id="cogvlm-chat-17b"
    )  # can be replaced by your model id like cogagent-chat-18b
    qwen_model_card = ModelCard(
        id="Qwen-VL-Chat"
    )  # can be replaced by your model id like cogagent-chat-18b
    return ModelList(data=[cogvlm_model_card, qwen_model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")
    
    if "qwen" in request.model:
        gen_params = dict(
            messages=request.messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 1024,
            echo=False,
            stream=request.stream,
        )
        response = generate_qwen(model, tokenizer, gen_params)
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


def generate_cogvlm(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict
):
    """
    Generates a response using the CogVLM model. It processes the chat history and image data, if any,
    and then invokes the model to generate a response.
    """

    for response in generate_stream_cogvlm(model, tokenizer, params):
        pass
    return response


def generate_qwen(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict
):
    """
    Generates a response using the Qwen-VL model. It processes the chat history and image data, if any,
    and then invokes the model to generate a response.
    """

    for response in generate_stream_qwen(model, tokenizer, params):
        pass
    return response


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


def process_qwen_history(
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
    if not os.path.exists('.tmp/'):
        os.makedirs('.tmp/')

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
                        image_path = f".tmp/{len(image_list)}.png"
                        image.save(image_path)
                        image_list.append(image_path)

        if role == "user":
            image_query = ""
            for idx, image_path in enumerate(image_list):
                image_query += f"Picture {idx + 1}: <img>{image_path}</img>\n"
            if i == len(messages) - 1:  # 最后一条用户消息
                last_user_query = image_query + text_content
            else:
                if formatted_history:
                    if formatted_history[-1][1] == "":
                        formatted_history[-1] = (formatted_history[-1][0], text_content)
                else:
                    formatted_history.append((image_query + text_content, ""))
            image_list = []
        elif role == "assistant":
            if formatted_history:
                if formatted_history[-1][1] != "":
                    assert (
                        False
                    ), f"the last query is answered. answer again. {formatted_history[-1][0]}, {formatted_history[-1][1]}, {text_content}"
                formatted_history[-1] = (formatted_history[-1][0], text_content)
            else:
                assert False, f"assistant reply before user"
        elif role == "system":
            if i == len(messages) - 1:
                last_user_query += text_content
            else:
                formatted_history.append((text_content, ""))
        else:
            assert False, f"unrecognized role: {role}"

    return last_user_query, formatted_history


@torch.inference_mode()
def generate_stream_cogvlm(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict
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

    if len(image_list) > 0:
        input_by_model = model.build_conversation_input_ids(
            tokenizer, query=query, history=history, images=[image_list[-1]]
        )
        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(DEVICE),
            "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(DEVICE),
            "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(DEVICE),
            "images": [[input_by_model["images"][0].to(DEVICE).to(torch_type)]],
        }
        if "cross_images" in input_by_model and input_by_model["cross_images"]:
            inputs["cross_images"] = [
                [input_by_model["cross_images"][0].to(DEVICE).to(torch_type)]
            ]

        input_echo_len = len(inputs["input_ids"][0])
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = {
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "do_sample": True if temperature > 1e-5 else False,
            "top_p": top_p if temperature > 1e-5 else 0,
            "streamer": streamer,
        }
        if temperature > 1e-5:
            gen_kwargs["temperature"] = temperature

        total_len = 0
        generated_text = ""
        with torch.no_grad():
            model.generate(**inputs, **gen_kwargs)
            for next_text in streamer:
                generated_text += next_text
                yield {
                    "text": generated_text,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": total_len - input_echo_len,
                        "total_tokens": total_len,
                    },
                }
        ret = {
            "text": generated_text,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
            },
        }
    else:
        generated_text = "An image input is required!"
        ret = {
            "text": generated_text,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    yield ret


@torch.inference_mode()
def generate_stream_qwen(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict
):
    """
    Generates a stream of responses using the Qwen-VL model in inference mode.
    It's optimized to handle continuous input-output interactions with the model in a streaming manner.
    """
    messages = params["messages"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    query, history = process_qwen_history(messages)
    input_echo_len = max_new_tokens + len(query)

    logger.debug(f"==== request ====\n{query}")

    gen_kwargs = {
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p if temperature > 1e-5 else 0,
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    total_len = 0
    generated_text = ""
    with torch.no_grad():
        streamer = model.chat_stream(tokenizer, query=query, history=history, **gen_kwargs)
        for next_text in streamer:
            generated_text = next_text
            yield {
                "text": generated_text,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": total_len - input_echo_len,
                    "total_tokens": total_len,
                },
            }
    ret = {
        "text": generated_text,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len,
        },
    }
    yield ret


gc.collect()
torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="OpenAI API Local Server",
        description="",
    )
    parser.add_argument('--model_path', default="THUDM/cogvlm-chat-hf")
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-q', '--quant', default=False, action='store_true')
    parser.add_argument('--tokenizer_path', default="lmsys/vicuna-7b-v1.5")
    parser.add_argument('--model_name', default="cogvlm")
    args = parser.parse_args()

    if "qwen" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )
    elif "cog" in args.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )
    else:
        raise ValueError(f"model {args.model_name} is not supported!")

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch_type = torch.bfloat16
    else:
        torch_type = torch.float16

    print(
        "========Use torch type as:{} with device:{}========\n\n".format(
            torch_type, args.device
        )
    )

    if "cuda" in args.device:
        if args.quant:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                load_in_4bit=True,
                trust_remote_code=True,
                torch_dtype=torch_type,
                low_cpu_mem_usage=True,
            ).eval()
        else:
            model = (
                AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    load_in_4bit=False,
                    trust_remote_code=True,
                    torch_dtype=torch_type,
                    low_cpu_mem_usage=True,
                )
                .to(args.device)
                .eval()
            )

    else:
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model_path, trust_remote_code=True
            )
            .float()
            .to(args.device)
            .eval()
        )
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)
