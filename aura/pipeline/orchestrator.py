from aura.pipeline.registry import STAGE
import yaml


class Orchestrator:

    def run(self, recipe_name):

        with open("config/default.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        with open("recipes/research/recipes.yaml", "r", encoding='utf-8') as f:
            y = yaml.safe_load(f)
            stages = y[recipe_name]["stages"]

            context = { }

            for stage_name in stages:
                stage = STAGE[stage_name](config)
                stage.run(context)
        