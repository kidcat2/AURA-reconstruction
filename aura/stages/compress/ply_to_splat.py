from aura.stages.base import BaseStage
import os


class PlyToSplat(BaseStage):

    OUTPUT = "compress"

    def __init__(self, config):

        self.output_dir = os.path.join(config["data"]["run_dir"], self.OUTPUT)

    def run(self, context):
        print("PlyToSplat: start")

        self.input_dir = context["gaussians"]

        self.make_context(context)

        print("PlyToSplat: done")

    def make_context(self, context):
        context["splat"] = self.output_dir
