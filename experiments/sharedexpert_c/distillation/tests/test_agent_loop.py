from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from omegaconf import OmegaConf

from experiments.sharedexpert_c.distillation.agent_loop import DomainOPDSingleTurn
from experiments.sharedexpert_c.training.agent_loop import SharedExpertSingleTurn


class DomainAgentLoopTest(unittest.IsolatedAsyncioTestCase):
    def make_loop(self, key):
        loop = object.__new__(DomainOPDSingleTurn)
        loop.config = OmegaConf.create({"reward_model": {"reward_kwargs": {"mopd_config": {
            "domain_key": key, "domain_to_teacher": {"vision": "perception"},
        }}}})
        return loop

    async def test_preserves_top_level_domain_and_existing_output_fields(self):
        loop = self.make_loop("domain")
        output = SimpleNamespace(extra_fields={"prompt_truncated": 0})
        with patch.object(SharedExpertSingleTurn, "run", new=AsyncMock(return_value=output)) as base:
            result = await loop.run({"temperature": 1}, domain="vision", raw_prompt=[])
        self.assertEqual(result.extra_fields, {"prompt_truncated": 0, "domain": "vision"})
        self.assertEqual(base.await_count, 1)

    async def test_nested_metadata_copied_before_base_mutation(self):
        loop = self.make_loop("extra_info.domain")
        metadata = {"domain": "vision", "index": 17}
        async def generate(self, params, **kwargs):
            kwargs["extra_info"]["index"] = 999
            return SimpleNamespace(extra_fields={})
        with patch.object(SharedExpertSingleTurn, "run", new=generate):
            result = await loop.run({}, extra_info=metadata, raw_prompt=[])
        self.assertEqual(result.extra_fields["extra_info"], {"domain": "vision", "index": 17})

    async def test_missing_domain_fails_before_generation(self):
        loop = self.make_loop("domain")
        with patch.object(SharedExpertSingleTurn, "run", new=AsyncMock()) as base:
            with self.assertRaises(ValueError):
                await loop.run({}, raw_prompt=[])
        base.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
