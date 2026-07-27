from aura.pipeline.stages import STAGE
import yaml

class Orchestrator:

    def run(self, recipe_name):

        with open("config/default.yaml", 'r') as f:
            config = yaml.safe_load(f)

        with open("recipes/recipes.yaml", "r") as f:
            y = yaml.safe_load(f)
            stages = y[recipe_name]["stages"]
            # TODO: 임시 - frame_extract/colmap 스킵 시 context 수동 설정. 전 단계 정상 실행 시 아래 두 줄 삭제하고 context = {} 로 복구
            # context = {
            #     "frames_dir": config["data"]["frames_dir"],
            #     "sparse": config["data"]["sfm_dir"],
            # }
            #context = {}

            context = {
                "frames_dir": config["data"]["frames_dir"],
                "sparse": config["data"]["sfm_dir"],
            }

            for stage_name in stages:
                stage = STAGE[stage_name](config)
                stage.run(context)
        