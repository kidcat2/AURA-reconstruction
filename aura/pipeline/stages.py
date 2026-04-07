from aura.stages.preprocess.frame_extract import FrameExtract
from aura.stages.sfm.colmap import Colmap
from aura.stages.reconstruction.gaussian_splatting import GaussianSplatting
from aura.stages.postprocess.ply_to_splat import PlyToSplat

STAGE = {
    "frame_extract" : FrameExtract,
    "colmap" : Colmap,
    "gaussian_splatting" : GaussianSplatting,
    "ply_to_splat" : PlyToSplat,
}
