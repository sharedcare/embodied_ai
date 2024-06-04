from typing import Dict, Optional, Union
import io
import re
import random
import cv2
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import matplotlib.colors as mcolors

from autogen import (
    Agent,
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
    config_list_from_json,
)
from vision_agent import VisionAgent
from robot_agent import RobotAgent
from chainlit import user_session
from chainlit.input_widget import Select, Switch, Slider
import chainlit as cl


async def ask_helper(func, **kwargs):
    res = await func(**kwargs).send()
    while not res:
        res = await func(**kwargs).send()
    return res


class ChainlitVisionAgent(VisionAgent):
    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ) -> bool:
        model_name = user_session.get("SETTINGS")["model"]
        labels = None
        if "qwen" in model_name.lower():
            labels = re.findall(r"<ref>[^<>]+<\/ref>", message)
            labels = [
                label.replace("<ref>", "").replace("</ref>", "") for label in labels
            ]
            coords = re.findall(r"<box>[^<>]+<\/box>", message)
            coords = [
                coord.replace("<box>", "")
                .replace("</box>", "")
                .replace("(", "")
                .replace(")", "")
                for coord in coords
            ]
            coords = [coord.split(",") for coord in coords]
        elif "cog" in model_name.lower():
            coords = re.findall(r"\[\[\d+,\d+,\d+,\d+\]\]", message)
            coords = [coord.replace("[[", "").replace("]]", "") for coord in coords]
            coords = [coord.split(",") for coord in coords]
        else:
            coords = []
        # Retrive input image
        img_path = user_session.get("IMAGE").path
        # read input image
        img = read_image(img_path)
        img_bytes = None
        boxes = []
        colors = []
        _, height, width = img.shape
        for coord in coords:
            # bounding box is [x0, y0, x1, y1]
            box = [(int(pos) / 1000) for pos in coord]
            box[0] *= width
            box[1] *= height
            box[2] *= width
            box[3] *= height
            box = torch.tensor(box)
            box = box.unsqueeze(0)
            boxes.append(box)
            colors.append(
                random.choice([_ for _ in mcolors.TABLEAU_COLORS.values()])
            )  # init color

        font_size = width // 30
        line_width = width // 100
        font_path = "SimSun.ttf"
        if len(boxes) > 0:
            boxes = torch.cat(boxes)
            # draw bounding box and fill color
            img = draw_bounding_boxes(
                img,
                boxes,
                labels=labels,
                width=line_width,
                font=font_path,
                font_size=font_size,
                colors=colors,
                fill=True,
            )
            # transform image to PIL image
            pil_img = torchvision.transforms.ToPILImage()(img)
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, "JPEG")

            image = cl.Image(
                name="bbox_image", content=img_bytes.getvalue(), display="inline"
            )
            cl.run_sync(
                cl.Message(
                    content=f'*Sending message to "{recipient.name}":*\n\n{message}',
                    author="VisionAgent",
                    elements=[image],
                ).send()
            )
        else:
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


class ChainlitRobotAgent(RobotAgent):
    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ) -> None:
        model_name = user_session.get("SETTINGS")["model"]
        labels = None
        if "qwen" in model_name.lower():
            labels = re.findall(r"<ref>[^<>]+<\/ref>", message)
            labels = [
                label.replace("<ref>", "").replace("</ref>", "") for label in labels
            ]
            coords = re.findall(r"<box>[^<>]+<\/box>", message)
            coords = [
                coord.replace("<box>", "")
                .replace("</box>", "")
                .replace("(", "")
                .replace(")", "")
                for coord in coords
            ]
            coords = [coord.split(",") for coord in coords]
        elif "cog" in model_name.lower():
            coords = re.findall(r"\[\[\d+,\d+,\d+,\d+\]\]", message)
            coords = [coord.replace("[[", "").replace("]]", "") for coord in coords]
            coords = [coord.split(",") for coord in coords]
        else:
            labels = []
            coords = []

        grasp_poses = {}
        # Execute plan
        for label in labels:
            grasp_poses[label] = self.get_grasp_pose(coords[label])

        res_object = cl.run_sync(
            ask_helper(
                cl.AskActionMessage(
                    content="Select object to grasp!",
                    actions=[
                        cl.Action(name="object", value=label, label=label)
                        for label in labels
                    ],
                )
            )
        )
        res_where = cl.run_sync(
            ask_helper(
                cl.AskActionMessage(
                    content="Select where to place the object!",
                    actions=[
                        cl.Action(name="where", value=label, label=label)
                        for label in labels
                    ],
                )
            )
        )
        if res_object.get("object") is not None and res_where.get("where") is not None:
            object_to_grasp = res_object.get("object")
            where_to_place = res_where.get("where")
            success = self.ur5_excute(
                object_to_grasp,
                grasp_poses[object_to_grasp],
                grasp_poses[where_to_place],
            )
            print(success)
        super(ChainlitRobotAgent, self).send(
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

    # With grounding
    # with_grounding = False
    # user_session.set("WITH_GROUNDING", with_grounding)

    config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

    models = [
        "llava-1.5",
        "cogvlm-chat",
        "cogagent-chat",
        "cogvlm-grounding-generalist",
        "qwen-vl-chat",
    ]

    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="MultiModal Models",
                values=models,
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
            # Switch(
            #     id="with_grounding",
            #     label="CogVLM/Qwen-VL - with grounding",
            #     inital=False,
            # ),
        ]
    ).send()

    vision_planner = ChainlitVisionAgent(
        name="Vision Planner",
        system_message="""
        You are a Vision Planner. You are supposed to suggest a plan with steps based on the vision inputs for robot agent to execute.
        
        When you receive a task, follow these steps:
            1. Identify the User's Intent: Understand the task and its context. The intent might involve moving the robot arm, manipulating objects, or gathering information.
            2. You come up with a logical plan to achieve the task. Your message must be a step by step sequential plan. If any visible object mentioned in the plan, you should include position with grounding.
        
        You send your plan to Robot Agent, who runs each sub task one by one.
        """,
        human_input_mode="NEVER",
        llm_config={"config_list": config_list},
    )
    user_proxy = ChainlitUserProxyAgent(
        "Human Proxy",
        system_message="Human Proxy. Only receives messages from Vision Planner and not particuss the cipates in the planning.",
        code_execution_config={
            "work_dir": "workspace",
            "use_docker": False,
        },
    )
    robot_agent = ChainlitRobotAgent(
        "Robot Agent",
        system_message="Robot Agent. Receive the plan and execute actions step by step based on the plan.",
        human_input_mode="NEVER",
        code_execution_config=False,
        llm_config=None,
    )

    # Create group chat
    groupchat = GroupChat(
        agents=[user_proxy, vision_planner, robot_agent], messages=[], max_round=50
    )
    manager = GroupChatManager(
        groupchat=groupchat, llm_config={"config_list": config_list}
    )

    user_session.set("VISION_PLANNER", vision_planner)
    user_session.set("USER_PROXY", user_proxy)
    user_session.set("ROBOT_AGENT", robot_agent)
    user_session.set("GROUPCHAT_MANAGER", manager)
    user_session.set("SETTINGS", settings)


@cl.on_settings_update
async def setup_agent(settings):
    config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
    for k, v in settings.items():
        config_list[0][k] = v

    print(config_list)
    vision_planner = ChainlitVisionAgent(
        name="Vision Planner",
        system_message="""
        You are a Vision Planner. You are supposed to suggest a plan with steps based on the vision inputs for robot agent to execute.
        
        When you receive a task, follow these steps:
            1. Identify the User's Intent: Understand the task and its context. The intent might involve moving the robot arm, manipulating objects, or gathering information.
            2. You come up with a logical plan to achieve the task. Your message must be a step by step sequential plan. If any visible object mentioned in the plan, you should include position with grounding.
        
        You send your plan to Robot Agent, who runs each sub task one by one.
        """,
        human_input_mode="NEVER",
        llm_config={"config_list": config_list},
    )
    user_proxy = ChainlitUserProxyAgent(
        "Human Proxy",
        system_message="Human Proxy. Only receives messages from Vision Planner and not particuss the cipates in the planning.",
        code_execution_config={
            "work_dir": "workspace",
            "use_docker": False,
        },
    )
    robot_agent = ChainlitRobotAgent(
        "Robot Agent",
        system_message="Robot Agent. Receive the plan and execute actions step by step based on the plan.",
        human_input_mode="NEVER",
        code_execution_config=False,
        llm_config=None,
    )

    # Create group chat
    groupchat = GroupChat(
        agents=[user_proxy, vision_planner, robot_agent], messages=[], max_round=50
    )
    manager = GroupChatManager(
        groupchat=groupchat, llm_config={"config_list": config_list}
    )

    # if settings["model"] in [
    #     "cogagent-chat",
    #     "cogvlm-grounding-generalist",
    #     "qwen-vl-chat",
    # ]:
    #     with_grounding = settings["with_grounding"]
    # else:
    #     with_grounding = False
    # user_session.set("WITH_GROUNDING", with_grounding)
    user_session.set("VISION_PLANNER", vision_planner)
    user_session.set("USER_PROXY", user_proxy)
    user_session.set("ROBOT_AGENT", robot_agent)
    user_session.set("GROUPCHAT_MANAGER", manager)
    user_session.set("SETTINGS", settings)


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

    # Retrieve Robot Agent
    robot_agent = user_session.get("ROBOT_AGENT")
    # Capture the video frame
    frame = robot_agent.get_image()
    if frame is not None:
        cv2.imwrite(".tmp.png", frame)
        image = cl.Image(name="tmp_image", path=".tmp.png")
        images = [image]

    # Retrieve Vision Agent
    vision_planner = user_session.get("VISION_PLANNER")

    # Retrive User Proxy
    user_proxy = user_session.get("USER_PROXY")

    # Retrive GroupChat Manager
    manager = user_session.get("GROUPCHAT_MANAGER")

    # Retrive with grounding property
    # with_grounding = user_session.get("WITH_GROUNDING")
    # new_token = "(with grounding)" if with_grounding else ""
    new_token = ""

    if len(images) >= 1:
        # Set input with image
        prompt = f"{message.content}{new_token}<img {images[-1].path}>"
        user_session.set("IMAGE", images[-1])
    else:
        # Set input without image
        prompt = message.content + new_token

    await cl.Message(
        content=f"Vision agent working on task: '{message.content}.'",
        elements=images,
    ).send()

    await cl.make_async(user_proxy.initiate_chat)(
        manager,
        message=prompt,
        clear_history=False,
    )
