"""Selected numerical kernels, without model loading or benchmark claims."""
from .losses import retrieval_loss, semantic_loss, perceptual_loss, total_loss
from .pgd import PGDConfig, PGDResult, optimize_image
from .verification import VerificationDecision, reasoning_divergence, decide_memory
from .diffusion import forward_noise, ddim_step

__version__ = "0.1.0"
__all__ = [
    "retrieval_loss", "semantic_loss", "perceptual_loss", "total_loss",
    "PGDConfig", "PGDResult", "optimize_image", "VerificationDecision",
    "reasoning_divergence", "decide_memory", "forward_noise", "ddim_step",
]
