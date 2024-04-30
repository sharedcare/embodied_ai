# Embodied AI Demo with AutoGen and Chainlit
### Requirements
```
pip install pyautogen chainlit
```
## Intro

This project is a demo of multimodal embodied AI with AutoGen.

## Get Started

1. Create a `.env` file as LLM config (using `.env.sample` as an example)
2. Change `base_url` in `.env` file to your LLM API url. By default, you should use `https://127.0.0.1:8001`.
3. Launch our multimodal model in an OpenAI API service by
```
python openai_api.py
```
4. Launch Chainlit web UI to start your conversations
```
chainlit run embodied_ai.py
```