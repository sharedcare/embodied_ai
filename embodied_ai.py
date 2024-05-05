from typing import Dict, Optional, Union

from autogen import Agent, AssistantAgent, UserProxyAgent, config_list_from_json
from vision_agent import VisionAgent
from chainlit import user_session
from chainlit.input_widget import Select, Switch, Slider
import chainlit as cl


async def ask_helper(func, **kwargs):
    res = await func(**kwargs).send()
    while not res:
        res = await func(**kwargs).send()
    return res


class RoboProxy(Agent):
    """Robot Proxy Agent"""
    def get_image():
        """Get RGB image data from the robot's camera
        """
        pass

    def get_depth():
        """Get depth data from the robot's camera
        """
        pass


class ChainlitVisionAgent(VisionAgent):
    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ) -> bool:
        cl.run_sync(
            cl.Message(
                content=f'*Sending message to "{recipient.name}":*\n\n{message}',
                author="VisionAgent",
            ).send()
        )
        super(VisionAgent, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )


class ChainlitUserProxyAgent(UserProxyAgent):
    def get_human_input(self, prompt: str) -> str:
        if prompt.startswith(
            "Provide feedback to assistant. Press enter to skip and use auto-reply"
        ):
            res = cl.run_sync(
                ask_helper(
                    cl.AskActionMessage,
                    content="Continue or provide feedback?",
                    actions=[
                        cl.Action(
                            name="continue", value="continue", label="✅ Continue"
                        ),
                        cl.Action(
                            name="feedback",
                            value="feedback",
                            label="💬 Provide feedback",
                        ),
                        cl.Action(
                            name="exit", value="exit", label="🔚 Exit Conversation"
                        ),
                    ],
                )
            )
            if res.get("value") == "continue":
                return ""
            if res.get("value") == "exit":
                return "exit"

        reply = cl.run_sync(ask_helper(cl.AskUserMessage, content=prompt, timeout=1e6))

        return reply["output"].strip()

    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        cl.run_sync(
            cl.Message(
                content=f'*Sending message to "{recipient.name}"*:\n\n{message}',
                author="UserProxyAgent",
            ).send()
        )
        super(ChainlitUserProxyAgent, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )


@cl.on_chat_start
async def on_chat_start():
    # Message history
    message_history = []
    user_session.set("MESSAGE_HISTORY", message_history)

    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="MultiModal Models",
                values=[
                    "llava-1.5-7b-hf",
                    "cogvlm-chat-17b",
                    "cogagent-chat-17b",
                ],
                initial_index=0,
            ),
            Slider(
                id="temperature",
                label="Temperature",
                initial=0.8,
                min=0,
                max=2,
                step=0.1,
            ),
            Slider(
                id="max_tokens",
                label="Max New Tokens",
                initial=256,
                min=0,
                max=1024,
                step=2,
            ),
        ]
    ).send()

    config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
    vision_assistant = ChainlitVisionAgent(
        name="vision_assistant",
        system_message="A vision assistant",
        llm_config={"config_list": config_list},
    )
    user_proxy = ChainlitUserProxyAgent(
        "user_proxy",
        code_execution_config={
            "work_dir": "workspace",
            "use_docker": False,
        },
    )

    user_session.set("VISION_AGENT", vision_assistant)
    user_session.set("USER_PROXY", user_proxy)

@cl.on_settings_update
async def setup_agent(settings):
    config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
    for k, v in settings.items():
        config_list[0][k] = v
    
    print(config_list)
    vision_assistant = ChainlitVisionAgent(
        name="vision_assistant",
        system_message="A vision assistant",
        llm_config={"config_list": config_list},
    )
    user_proxy = ChainlitUserProxyAgent(
        "user_proxy",
        code_execution_config={
            "work_dir": "workspace",
            "use_docker": False,
        },
    )

    user_session.set("VISION_AGENT", vision_assistant)
    user_session.set("USER_PROXY", user_proxy)

# On message
@cl.on_message
async def main(message: cl.Message):
    # Send empty message for loading
    msg = cl.Message(
        content=f"",
        author="Vision Chat",
    )
    await msg.send()

    # Processing images (if any)
    images = [file for file in message.elements if "image" in file.mime]

    # Retrieve Vision Agent
    vision_agent = user_session.get("VISION_AGENT")

    # Retrive User Proxy
    user_proxy = user_session.get("USER_PROXY")

    if len(images) >= 1:
        # Set input with image
        prompt = f"{message.content}<img {images[-1].path}>"
    else:
        # Set input without image
        prompt = message.content

    await cl.Message(
        content=f"Vision agent working on task: {message.content}."
    ).send()

    await cl.make_async(user_proxy.initiate_chat)(
        vision_agent,
        message=prompt,
    )
