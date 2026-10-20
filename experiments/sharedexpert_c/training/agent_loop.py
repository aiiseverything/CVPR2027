"""Single-turn rollout with multimodal-safe prompt truncation.

Only ordinary user text is shortened. Image placeholders, image features and the
Qwen chat template remain intact. The response is sampled once (no forced suffix
or off-policy continuation) and hard-capped at the configured 8192 tokens.
"""

from copy import deepcopy

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop


class SharedExpertSingleTurn(SingleTurnAgentLoop):
    def _truncate_prompt_ids(self, prompt_ids):
        # Base.apply_chat_template calls this before returning. Do not slice
        # image token spans; apply_chat_template below handles text instead.
        return prompt_ids

    async def apply_chat_template(self, messages, **kwargs):
        messages = deepcopy(messages)
        limit = self._get_prompt_length()
        ids = await super().apply_chat_template(messages, **kwargs)
        self.original_prompt_length = len(ids)
        if limit is None or len(ids) <= limit:
            return ids

        blocks = []
        for message in messages:
            if message.get("role") != "user":
                continue
            if isinstance(message.get("content"), str):
                blocks.append((message, "content"))
            else:
                for block in message.get("content", []):
                    if block.get("type") == "text":
                        blocks.append((block, "text"))

        for _ in range(64):
            candidates = [
                (self.tokenizer.encode(block[key], add_special_tokens=False), block, key)
                for block, key in blocks
            ]
            candidates = [entry for entry in candidates if len(entry[0]) > 8]
            if not candidates:
                raise ValueError(
                    f"Image/template tokens cannot fit prompt limit {limit}; "
                    "reduce image pixels or inspect this example."
                )
            tokens, block, key = max(candidates, key=lambda entry: len(entry[0]))
            keep = max(8, len(tokens) - (len(ids) - limit) - 8)
            front = max(1, 3 * keep // 4)
            # Keep both the question start and final output-format instruction.
            block[key] = self.tokenizer.decode(tokens[:front]) + "\n" + self.tokenizer.decode(tokens[-(keep-front):])
            ids = await super().apply_chat_template(messages, **kwargs)
            if len(ids) <= limit:
                return ids
        raise ValueError("Unable to fit text prompt in 64 deterministic truncation rounds")

    async def run(self, sampling_params, **kwargs):
        kwargs = dict(kwargs)
        kwargs["raw_prompt"] = deepcopy(kwargs["raw_prompt"])
        image_blocks = []
        for message in kwargs["raw_prompt"]:
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "image":
                        image_blocks.append(block)
        # Qwen's patch16/merge2 consumes approximately one token per 1024
        # pixels. Bound the combined image budget, including multi-image ViRL.
        pixel_cap = min(786432, 1572864 // max(1, len(image_blocks)))
        for block in image_blocks:
            block["min_pixels"] = min(int(block.get("min_pixels", 4096)), pixel_cap)
            block["max_pixels"] = min(int(block.get("max_pixels", 786432)), pixel_cap)
        output = await super().run(sampling_params, **kwargs)
        output.extra_fields["original_prompt_tokens"] = self.original_prompt_length
        output.extra_fields["prompt_truncated"] = int(self.original_prompt_length > len(output.prompt_ids))
        if len(output.prompt_ids) > self.config.data.max_prompt_length:
            raise AssertionError("Prompt limit invariant violated")
        return output
