# Copyright 2026 The SeededGrasp Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import base64
import json
import os
import re
from abc import ABC, abstractmethod

import numpy as np
import plotly.io as pio
from dotenv import load_dotenv
from openai import OpenAI

pio.renderers.default = "browser"


class VLM(ABC):
    """
    Abstract base class for VLM models.
    """

    @abstractmethod
    def load_img(self, img_path):
        """
        Takes in img path and loads it in required format for model.
        """

    @abstractmethod
    def prompt(self, msg):
        """
        Takes in user prompt, sends to VLM, returns output.
        """

    @abstractmethod
    def rate_limited(self):
        """
        Check if api key is rate limited (true or false).
        """

    @abstractmethod
    def get_model_id(self):
        """
        Return str identifying the model.
        """


class GeminiOpenRouter(VLM):
    def __init__(self, model: str = None):
        super().__init__()
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # Available models
        self.model = "google/gemini-3-flash-preview"
        if model is not None:
            self.model = model
        # self.model = "google/gemini-3.1-flash-lite-preview"

        # Thinking level
        # self.thinking_level = "minimal"
        # self.thinking_level = "low"
        self.thinking_level = "medium"
        # self.thinking_level = "high"

    def prompt(self, msg):
        completion = self.client.chat.completions.create(
            extra_body={
                "provider": {
                    "only": ["google-vertex"],  # Only use the 'google' slug
                    "allow_fallbacks": False,  # Do not switch if Google is down
                }
            },
            model=self.model,
            reasoning_effort=self.thinking_level,
            messages=msg,
        )
        return completion

    def load_img(self, img_path):
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def get_model_id(self):
        return self.model

    def rate_limited(self):
        return False


class QwenOpenRouter(VLM):
    def __init__(self, model: str = None):
        super().__init__()
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # Available models
        # self.model = "qwen/qwen3.5-397b-a17b"
        self.model = "qwen/qwen3.5-flash-02-23"
        if model is not None:
            self.model = model
        # self.model = "qwen/qwen3.5-9b"
        # self.model = "qwen/qwen3.7-flash"

        # Thinking level
        self.thinking_level = "minimal"
        # self.thinking_level = "low"
        # self.thinking_level = "medium"
        # self.thinking_level = "high"

    def prompt(self, msg):
        completion = self.client.chat.completions.create(
            extra_body={"provider": {"only": ["alibaba"], "allow_fallbacks": False}},
            model=self.model,
            reasoning_effort=self.thinking_level,
            messages=msg,
        )
        return completion

    def load_img(self, img_path):
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def get_model_id(self):
        return self.model

    def rate_limited(self):
        return False


class GPTOpenRouter(VLM):
    def __init__(self, model: str = None):
        super().__init__()
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # Available models
        # self.model = "openai/gpt-5.4-nano"
        self.model = "openai/gpt-5.4"
        if model is not None:
            self.model = model
        # self.model = "openai/gpt-5.6-luna"
        # self.model = "openai/gpt-5.6-sol"

        # Thinking level
        self.thinking_level = "minimal"
        # self.thinking_level = "low"
        # self.thinking_level = "medium"
        # self.thinking_level = "high"

    def prompt(self, msg):
        completion = self.client.chat.completions.create(
            model=self.model,
            reasoning_effort=self.thinking_level,
            messages=msg,
        )
        return completion

    def load_img(self, img_path):
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def get_model_id(self):
        return self.model

    def rate_limited(self):
        return False


class ClaudeOpenRouter(VLM):
    def __init__(self, model: str = None):
        super().__init__()
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # Available models
        # self.model = "anthropic/claude-sonnet-4.6"
        # self.model = "anthropic/claude-opus-4.6"
        self.model = "anthropic/claude-opus-4.8"
        if model is not None:
            self.model = model
        # self.model = "anthropic/claude-sonnet-5"

        # Thinking level
        # self.thinking_level = "minimal"
        # self.thinking_level = "low"
        # self.thinking_level = "medium"
        self.thinking_level = "high"

    def prompt(self, msg):
        completion = self.client.chat.completions.create(
            extra_body={
                "provider": {
                    "only": ["anthropic"],
                    "allow_fallbacks": False,
                }
            },
            model=self.model,
            reasoning_effort=self.thinking_level,
            messages=msg,
        )
        return completion

    def load_img(self, img_path):
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def get_model_id(self):
        return self.model

    def rate_limited(self):
        return False


class Message(ABC):
    """
    Abstract Base Class for prompt message templates
    """

    @abstractmethod
    def construct_msg(self, input):
        """
        Construct ready-to-send message based on whatever input.
        """
        pass

    @abstractmethod
    def get_version_id(self):
        """
        Return version id of current message template
        """
        pass

    @abstractmethod
    def post_process(self, img, pcd, response_json):
        """
        Return dict of additional values to store in master json entry.
        """


class BatchedPromptV3_1(Message):
    def __init__(self):
        self.version_id = "3.1"
        self.system_prompt = [
            "You are a robotic manipulation specialist. Please help determine a stable grasp location for the specified object in the provided image of each scene. The robot gripper will attempt to grab the object at that exact location.",
            "Express the desired grasp location as a pixel coordinate (y, x) normalized to the 0-1000 range.",
            "Output the result as a JSON in the following format:",
            "Example JSON result: {scene_number: {object: object_description, coords: (y, x), explanation: brief_justification}}.",
            "Critical Requirements:",
            "1. Ensure that the grasp location is free from collisions with surrounding objects.",
            "2. Ensure that the location is on the surface of the object as it will be projected onto the 3D point cloud of the scene. The location must be on the target object, where the gripper will grasp, it cannot be in empty space for hollow objects.",
            "3. Ensure that the gripper can encapsulate the object around the chosen grasp location. Grippers are standard sizes and cannot fully wrap around large objects. In such cases, edges or outcrops may be a better grasping location, do not attempt to grab the entire object body.",
            "4. Ensure that your output follows the specified JSON format.",
        ]

    def construct_msg(self, input):
        """
        For this message input should be a list of object names and images.
        """
        msg = [
            {"role": "system", "content": "\n".join(self.system_prompt)},
            {"role": "user", "content": []},
        ]

        for i, data in enumerate(input):
            msg[1]["content"].append(
                {"type": "text", "text": f"\n\nScene number {i+1}:\nObject description: {data[0]}."}
            )
            msg[1]["content"].append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data[1]}"}}
            )

        return msg

    def get_version_id(self):
        return self.version_id

    def post_process(self, img, pcd, response_json):
        h, w, c = np.array(img).shape
        seed_pt = pcd_px_coord(pcd, h, w, response_json["coords"]).tolist()
        return {
            "norm_coords": response_json["coords"],
            "seed_pt": seed_pt,
            "explanation": response_json["explanation"],
        }

    def extract_json(self, response):
        # Get the raw string from the response object
        raw_text = response.choices[0].message.content.strip()
        raw_text = re.sub(r"\((\s*\d+\s*,\s*\d+\s*)\)", r"[\1]", raw_text)

        # 1. Try to parse the entire string as JSON immediately
        # (Handles your second example format)
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        # 2. If that fails, try to find content between ```json and ```
        # (Handles your first example format)
        markdown_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
        if markdown_match:
            try:
                return json.loads(markdown_match.group(1))
            except json.JSONDecodeError:
                pass

        # 3. Last ditch effort: Try to find anything between the first { and last }
        # (Handles cases where the model puts text before or after the JSON)
        braces_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
        if braces_match:
            try:
                return json.loads(braces_match.group(1))
            except json.JSONDecodeError:
                pass

        return None


def unnorm_coords(h, w, norm_coords):
    coords = (round(h * (norm_coords[0] / 1000.0)), round(w * (norm_coords[1] / 1000.0)))
    return coords


def pcd_px_coord(pcd, h, w, norm_coords):
    """
    Projects a px coordinate from 2D image onto corresponding 3D PCD.
    """
    pcd = pcd.reshape((h, w, 3))
    norm_coords = unnorm_coords(h, w, norm_coords)
    return pcd[norm_coords[0]][norm_coords[1]]


# Helper functions
def set_nested(data, path, value):
    """
    data: the dictionary
    path: a list of keys e.g. ["users", "settings", "theme"]
    value: the value to set at the end of the path
    """
    for key in path[:-1]:
        # If key doesn't exist or isn't a dict, create an empty dict
        data = data.setdefault(key, {})
    data[path[-1]] = value


# Maps the --vlm choice to its class. The default model id inside each class is
# the one used for that family in the paper; Gemini is used for all reported
# results, the others for the VLM ablation.
VLM_FAMILIES = {
    "gemini": GeminiOpenRouter,
    "gpt": GPTOpenRouter,
    "qwen": QwenOpenRouter,
    "claude": ClaudeOpenRouter,
}


def build_vlm(family: str = "gemini", model: str = None) -> VLM:
    """Construct a VLM client by family name, optionally overriding the model id.

    Args:
        family: One of VLM_FAMILIES ("gemini", "gpt", "qwen", "claude").
        model: Explicit OpenRouter model id (e.g. "openai/gpt-5.6-luna").
               Defaults to the family's own default.
    """
    if family not in VLM_FAMILIES:
        raise ValueError(f"Unknown VLM family {family!r}; choose from {sorted(VLM_FAMILIES)}")
    return VLM_FAMILIES[family](model=model)
