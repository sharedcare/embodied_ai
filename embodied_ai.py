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


USER_PROXY_SYSTEM_MESSAGE = (
    "Human Proxy. Only receives messages from Vision Planner and not participates in the planning."
)
VISION_PLANNER_SYSTEM_MESSAGE = """
You are a Vision Planner. You are supposed to suggest a plan with steps based on the input image and allowed actions for robot agent to execute.

Allowed actions:
reach_to <object> (where): move arm to a position of an object
go_home: move arm to initial position
close_gripper: close hand gripper
open_gripper: open hand gripper
open <object>: open an object, it can be a door or a drawer
close <object>: close an object, it can be a door or a drawer
pick_up <object>: move arm to pick up an object
place <a_object> <b_object> (where): move arm to place object on hand to a position of another object
wait time(second): wait and do not move for certain time on second
TERMINATE: end of plan

(where) is a position of the object, you can only choose one of (above), (bottom), (front), (rear), (left), (right) and (center).

When you receive a task, follow these steps:
    1. Identify the User's Intent: Understand the task and its context. The intent might involve moving the robot arm, manipulating objects, or gathering information.
    2. You come up with a logical plan to achieve the task. Your message must be a step by step sequential plan in 3-10 steps and each step should start with [step] tag.

You send your plan to Robot Agent, who runs each sub task one by one.

For example,
--------------------
Your task:
Put a heated egg in the sink.\n
User's Intent:
The user wants the robot arm to heat a egg and put it in a sink.\n
Plan:
[step] pick_up <egg>
[step] reach_to <microwave> (front)
[step] open <microwave>
[step] place <egg> <microwave> (center)
[step] close <microwave>
[step] wait 120
[step] open <microwave>
[step] pick_up <egg>
[step] close <microwave>
[step] reach_to <sink> (above)
[step] place <egg> <sink> (center)
[step] TERMINATE
--------------------
Your task:"""
VISION_CRITIC_SYSTEM_MESSAGE = f"""
You are a Vision Critic. You are supposed to revise and double check plan from Vision Planner and provide feedback. Refine the plan if it is not correct or reasonable. Follow the instructions from Vision Planner:
'{VISION_PLANNER_SYSTEM_MESSAGE}'
"""
VISION_ASSISTANT_SYSTEM_MESSAGE = "You are a Vision Assistant. You are supposed to help Vision Planner to locate object in the image. If an object mentioned in the plan is visible in the input image, your message should output the bounding box with label"
ROBOT_AGENT_SYSTEM_MESSAGE = "Robot Agent. Receive the plan and execute actions step by step based on the plan."


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
        plan = message.split("[step]")[1:-1]
        plan = [step.strip() for step in plan]
        user_session.set("PLAN", plan)

        labels = re.findall(r"<.+?(?=>)>", message)
        labels = [label.replace("<", "").replace(">", "").strip() for label in set(labels)]
        objects = {key: None for key in labels}
        user_session.set("OBJECTS", objects)
        img_path = user_session.get("IMAGE").path

        if "qwen" in model_name.lower():
            obj_prompt = ""
            for label in labels:
                obj_prompt += label + ","
            prompt_msg = f"The location of {obj_prompt[:-1]}:<img {img_path}>"
        elif "cog" in model_name.lower():
            prompt_msg = f"Where is {labels[0]} using the format [[x1,y1,x2,y2]]."

        cl.run_sync(
            cl.Message(
                content=f"Plan:\n{message}",
                author="Vision Planner",
            ).send()
        )

        cl.run_sync(
            cl.Message(
                content=f'*Sending message to "{recipient.name}":*\n\n{prompt_msg}',
                author="Vision Planner",
            ).send()
        )

        super(ChainlitVisionAgent, self).send(
            message=prompt_msg,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )


class ChainlitVisionCritic(VisionAgent):
    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ) -> bool:
        model_name = user_session.get("SETTINGS")["model"]
        plan = message.split("[step]")[1:-1]
        plan = [step.strip() for step in plan]
        user_session.set("PLAN", plan)

        labels = re.findall(r"<.+?(?=>)>", message)
        labels = [
            label.replace("<", "").replace(">", "").strip() for label in set(labels)
        ]
        objects = {key: None for key in labels}
        user_session.set("OBJECTS", objects)
        img_path = user_session.get("IMAGE").path

        if "qwen" in model_name.lower():
            obj_prompt = ""
            for label in labels:
                obj_prompt += label + ","
            prompt_msg = f"The location of {obj_prompt[:-1]}:<img {img_path}>"
        elif "cog" in model_name.lower():
            prompt_msg = f"Where is {labels[0]} using the format [[x1,y1,x2,y2]]."

        cl.run_sync(
            cl.Message(
                content=f"Plan:\n{message}",
                author="Vision Critic",
            ).send()
        )

        cl.run_sync(
            cl.Message(
                content=f'*Sending message to "{recipient.name}":*\n\n{prompt_msg}',
                author="Vision Critic",
            ).send()
        )

        super(ChainlitVisionAgent, self).send(
            message=prompt_msg,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )


class ChainlitVisionAssistant(VisionAgent):
    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ) -> bool:
        model_name = user_session.get("SETTINGS")["model"]
        objects = user_session.get("OBJECTS")

        if "qwen" in model_name.lower():
            labels = re.findall(r"<ref>[^<>]+<\/ref>", message)
            labels = [label.replace("<ref>", "").replace("</ref>", "") for label in labels]
            coords = re.findall(r"<box>[^<>]+<\/box>", message)
            coords = [
                coord.replace("<box>", "").replace("</box>", "").replace("(", "").replace(")", "") for coord in coords
            ]
            coords = [coord.split(",") for coord in coords]
            for key, value in objects.items():
                if value is None:
                    objects[key] = coords[labels.index(key)]
        elif "cog" in model_name.lower():
            coords = re.findall(r"\[\[\d+,\d+,\d+,\d+\]\]", message)
            coords = [coord.replace("[[", "").replace("]]", "") for coord in coords]
            coords = [coord.split(",") for coord in coords]
            for key, value in objects.items():
                if value is None and key == message:
                    objects[key] = coords[0]
        else:
            coords = []

        prompt_msg = f"Detected objects: {[obj for obj in objects.keys()]}"
        for key, value in objects.items():
            if value is None:
                prompt_msg = f"Where is {key} using the format [[x1,y1,x2,y2]]."

        super(ChainlitVisionAssistant, self).send(
            message=prompt_msg,
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
        objects = user_session.get("OBJECTS")

        # Retrive input image
        img_path = user_session.get("IMAGE").path
        # read input image
        img = read_image(img_path)
        img_bytes = None
        labels = []
        boxes = []
        colors = []
        _, height, width = img.shape
        for obj, coord in objects.items():
            # bounding box is [x0, y0, x1, y1]
            box = [(int(pos) / 1000) for pos in coord]
            box[0] *= width
            box[1] *= height
            box[2] *= width
            box[3] *= height
            box = torch.tensor(box)
            box = box.unsqueeze(0)
            boxes.append(box)
            labels.append(obj)
            colors.append(random.choice([_ for _ in mcolors.TABLEAU_COLORS.values()]))  # init color

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

            image = cl.Image(name="bbox_image", content=img_bytes.getvalue(), display="inline")
            cl.run_sync(
                cl.Message(
                    content=f'*Sending message to "{recipient.name}":*\n\n{message}',
                    author="Robot Agent",
                    elements=[image],
                ).send()
            )
        else:
            cl.run_sync(
                cl.Message(
                    content=f'*Sending message to "{recipient.name}":*\n\n{message}',
                    author="Robot Agent",
                ).send()
            )

        grasp_poses = {}
        # Execute plan
        for label in labels:
            grasp_poses[label] = self.get_grasp_pose(objects[label])

        print(grasp_poses)

        res_object = cl.run_sync(
            ask_helper(
                cl.AskActionMessage,
                content="Select object to grasp!",
                actions=[cl.Action(name="object", value=label, label=label) for label in labels],
            )
        )
        res_where = cl.run_sync(
            ask_helper(
                cl.AskActionMessage,
                content="Select where to place the object!",
                actions=[cl.Action(name="where", value=label, label=label) for label in labels],
            )
        )
        if res_object.get("object") is not None and res_where.get("where") is not None:
            object_to_grasp = res_object.get("object")
            where_to_place = res_where.get("where")
            success = self.robot_execute(
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
        if prompt.startswith("Provide feedback to assistant. Press enter to skip and use auto-reply"):
            res = cl.run_sync(
                ask_helper(
                    cl.AskActionMessage,
                    content="Continue or provide feedback?",
                    actions=[
                        cl.Action(name="continue", value="continue", label="✅ Continue"),
                        cl.Action(
                            name="feedback",
                            value="feedback",
                            label="💬 Provide feedback",
                        ),
                        cl.Action(name="exit", value="exit", label="🔚 Exit Conversation"),
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


def state_transition(last_speaker: Agent, groupchat: GroupChat):
    """
    Routing: Human -> UserProxy -> VisionPlanner -> VisionCritic -> VisionAssistant -- located all Objects -- VisionAssistant -> RobotAgent
                                                                            |                                                          |
                                                                            |-------------- no Object need to be located --------------|
    """
    messages = groupchat.messages
    robot_agent = user_session.get("ROBOT_AGENT")
    vision_planner = user_session.get("VISION_PLANNER")
    vision_critic = user_session.get("VISION_CRITIC")
    vision_assistant = user_session.get("VISION_ASSISTANT")
    user_proxy = user_session.get("USER_PROXY")

    if last_speaker is user_proxy:
        return vision_planner
    elif last_speaker is vision_planner:
        return vision_critic
    elif last_speaker is vision_critic:
        objects = user_session.get("OBJECTS")
        if objects is None:
            return robot_agent
        else:
            return vision_assistant
    elif last_speaker is vision_assistant:
        objects = user_session.get("OBJECTS")
        for key, value in objects.items():
            if value is None:
                return vision_assistant
        return robot_agent
    elif last_speaker is robot_agent:
        if messages[-1]["content"] == "success":
            return user_proxy


@cl.on_chat_start
async def on_chat_start():
    # Message history
    message_history = []
    user_session.set("MESSAGE_HISTORY", message_history)

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
                initial=0.2,
                min=0,
                max=1,
                step=0.05,
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

    setup_agent(config_list)
    user_session.set("SETTINGS", settings)


@cl.on_settings_update
async def setup(settings):
    config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
    for k, v in settings.items():
        config_list[0][k] = v

    print(config_list)
    setup_agent(config_list)
    user_session.set("SETTINGS", settings)


def setup_agent(config):
    vision_planner = ChainlitVisionAgent(
        name="Vision Planner",
        system_message=VISION_PLANNER_SYSTEM_MESSAGE,
        human_input_mode="NEVER",
        llm_config={"config_list": config},
    )
    vision_critic = ChainlitVisionAgent(
        name="Vision Critic",
        system_message=VISION_CRITIC_SYSTEM_MESSAGE,
        human_input_mode="NEVER",
        llm_config={"config_list": config},
    )
    vision_assistant = ChainlitVisionAssistant(
        name="Vision Assistant",
        system_message=VISION_ASSISTANT_SYSTEM_MESSAGE,
        human_input_mode="NEVER",
        llm_config={"config_list": config},
    )
    user_proxy = ChainlitUserProxyAgent(
        "Human Proxy",
        system_message=USER_PROXY_SYSTEM_MESSAGE,
        code_execution_config={
            "work_dir": "workspace",
            "use_docker": False,
        },
    )
    robot_agent = ChainlitRobotAgent(
        "Robot Agent",
        system_message=ROBOT_AGENT_SYSTEM_MESSAGE,
        human_input_mode="NEVER",
        code_execution_config=False,
        llm_config=None,
    )

    # Create group chat
    groupchat = GroupChat(
        agents=[user_proxy, vision_planner, vision_critic, vision_assistant, robot_agent],
        messages=[],
        max_round=20,
        speaker_selection_method=state_transition,
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config={"config_list": config})

    user_session.set("VISION_PLANNER", vision_planner)
    user_session.set("VISION_CRITIC", vision_critic)
    user_session.set("VISION_ASSISTANT", vision_assistant)
    user_session.set("USER_PROXY", user_proxy)
    user_session.set("ROBOT_AGENT", robot_agent)
    user_session.set("GROUPCHAT_MANAGER", manager)


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

    if len(images) >= 1:
        # Set input with image
        prompt = f"{message.content}<img {images[-1].path}>"
        user_session.set("IMAGE", images[-1])
    else:
        # Set input without image
        prompt = message.content

    await cl.Message(
        content=f"Vision Planner working on task: '{message.content}.'",
        elements=images,
    ).send()

    await cl.make_async(user_proxy.initiate_chat)(
        manager,
        message=prompt,
        clear_history=False,
    )
