from aura.stages.frames.frame_extract import FrameExtract
from aura.stages.pose.colmap import Colmap
from aura.stages.train.gaussian_splatting import GaussianSplatting
from aura.stages.compress.ply_to_splat import PlyToSplat

STAGE = {
    "frame_extract" : FrameExtract,
    "colmap" : Colmap,
    "gaussian_splatting" : GaussianSplatting,
    "ply_to_splat" : PlyToSplat,
}
