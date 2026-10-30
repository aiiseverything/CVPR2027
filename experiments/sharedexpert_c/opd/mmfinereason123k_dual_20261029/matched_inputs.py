"""Preserve original RGB pixels until the shared Qwen processor resizes them."""
from copy import deepcopy
from io import BytesIO
from PIL import Image
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from experiments.sharedexpert_c.training.agent_loop import SharedExpertSingleTurn


class MatchedImageDataset(RLHFDataset):
    @classmethod
    async def process_vision_info(cls, messages, image_patch_size, config):
        images = []
        for message in messages:
            for block in message.get("content", []) if isinstance(message.get("content"), list) else []:
                if block.get("type") == "video":
                    raise ValueError("This frozen dataset contains images only")
                if block.get("type") != "image":
                    continue
                value = block.get("image", block.get("path"))
                if isinstance(value, Image.Image):
                    images.append(value.convert("RGB"))
                elif block.get("bytes") is not None:
                    with Image.open(BytesIO(block["bytes"])) as im:
                        images.append(im.convert("RGB"))
                else:
                    with Image.open(value) as im:
                        images.append(im.convert("RGB"))
        if len(images) != 1:
            raise ValueError(f"Expected exactly one original image, received {len(images)}")
        return images, None


class MatchedSingleTurn(SharedExpertSingleTurn):
    async def apply_chat_template(self, messages, **kwargs):
        ids = await SingleTurnAgentLoop.apply_chat_template(self, deepcopy(messages), **kwargs)
        self.original_prompt_length = len(ids)
        if self._get_prompt_length() is not None and len(ids) > self._get_prompt_length():
            raise ValueError("Frozen curation prompt exceeds training budget; refusing to truncate the question")
        return ids
