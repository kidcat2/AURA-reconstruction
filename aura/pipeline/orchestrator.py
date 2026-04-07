from aura.pipeline.stages import STAGE
import yaml

class Orchestrator:

    def run(self, recipe_name):

        with open("config/default.yaml", 'r') as f:
            config = yaml.safe_load(f)

        with open("recipes/recipes.yaml", "r") as f:
            y = yaml.safe_load(f)
            stages = y[recipe_name]["stages"]
            context = {}

            for stage_name in stages:
                stage = STAGE[stage_name]()
                stage.run(config, context)
        