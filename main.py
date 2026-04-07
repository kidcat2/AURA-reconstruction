from aura.pipeline.orchestrator import Orchestrator
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--recipe", default="colmap_3dgs")
    args = parser.parse_args()

    orchestrator = Orchestrator()
    orchestrator.run(args.recipe)
