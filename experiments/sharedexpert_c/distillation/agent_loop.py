"""Preserve explicit domain metadata through VERL's async rollout boundary."""

from copy import deepcopy

from experiments.sharedexpert_c.training.agent_loop import SharedExpertSingleTurn
from experiments.sharedexpert_c.distillation.routing import route_record


class DomainOPDSingleTurn(SharedExpertSingleTurn):
    async def run(self, sampling_params, **kwargs):
        config = self.config.reward_model.reward_kwargs.mopd_config
        # Resolve before generation: a missing domain must not waste a rollout.
        route_record(kwargs, config)
        root_key = config.get("domain_key", "domain").split(".", 1)[0]
        preserved = deepcopy(kwargs[root_key])
        output = await super().run(sampling_params, **kwargs)
        # The trainer moves arbitrary parquet columns into gen_batch and the
        # stock loop discards them. Returning this root restores the exact
        # metadata, including a configured path such as extra_info.domain.
        output.extra_fields[root_key] = preserved
        return output
