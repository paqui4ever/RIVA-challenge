from typing import Optional, Sequence, Tuple
from torchvision.models.detection.rpn import AnchorGenerator

_REFERENCE_TARGET_SIZE = 1008
_SAM3_TUNED_ANCHOR_SIZES = ((71, 78), (92, 104), (123, 135), (158, 168))
_SAM3_TUNED_ASPECT_RATIOS = ((0.82, 1.0, 1.12),) * 4
# SECOND BEST
#_CELL_DINO_TUNED_ANCHOR_SIZES = ((78, 85), (94, 104), (110, 127), (132, 140))
# BEST ANCHOR SIZES
_CELL_DINO_TUNED_ANCHOR_SIZES = ((83, 84), (94, 96), (110, 112), (115, 117))
#_CELL_DINO_TUNED_ANCHOR_SIZES = ((92, 93), (100, 102), (110, 112), (102, 104))
#  SECOND BEST ASPECT RATIOS
#_CELL_DINO_TUNED_ASPECT_RATIOS = ((0.825, 1.0, 1.115),) * 4
# BEST ASPECT RATIOS (very similar to second best)
_CELL_DINO_TUNED_ASPECT_RATIOS = ((0.825, 1.0, 1.05),) * 4
_LEGACY_CELL_DINO_ANCHOR_SIZES = ((16, 24, 32), (48, 64, 80), (96, 128, 160), (192, 256, 320))
_LEGACY_CELL_DINO_ASPECT_RATIOS = ((0.85, 1.0, 1.15),) * 4


def _validate_anchor_layout(
    sizes: Tuple[Tuple[int, ...], ...],
    aspect_ratios: Tuple[Tuple[float, ...], ...],
    num_levels: int,
) -> None:
    if len(sizes) != num_levels:
        raise ValueError(f"Expected {num_levels} anchor size levels, got {len(sizes)}")
    if len(aspect_ratios) != num_levels:
        raise ValueError(f"Expected {num_levels} aspect ratio levels, got {len(aspect_ratios)}")


def _coerce_anchor_sizes(sizes: Sequence[Sequence[int]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in level) for level in sizes)


def _coerce_anchor_aspect_ratios(
    aspect_ratios: Sequence[Sequence[float]],
) -> Tuple[Tuple[float, ...], ...]:
    return tuple(tuple(float(v) for v in level) for level in aspect_ratios)


def _scale_anchor_sizes(
    sizes: Tuple[Tuple[int, ...], ...],
    target_size: int,
    reference_target_size: int = _REFERENCE_TARGET_SIZE,
) -> Tuple[Tuple[int, ...], ...]:
    if target_size <= 0:
        raise ValueError(f"target_size must be > 0, got {target_size}")

    if target_size == reference_target_size:
        return sizes

    scale = float(target_size) / float(reference_target_size)
    return tuple(
        tuple(max(1, int(round(size * scale))) for size in level)
        for level in sizes
    )


def _build_cell_dino_anchor_generator(
    target_size: int,
    num_levels: int,
    anchor_profile: str = "sam3_tuned",
    anchor_sizes: Optional[Sequence[Sequence[int]]] = None,
    anchor_aspect_ratios: Optional[Sequence[Sequence[float]]] = None,
) -> AnchorGenerator:
    if anchor_sizes is not None:
        sizes = _coerce_anchor_sizes(anchor_sizes)
    elif anchor_profile == "sam3_tuned":
        sizes = _scale_anchor_sizes(_SAM3_TUNED_ANCHOR_SIZES, target_size)
    elif anchor_profile == "legacy":
        sizes = _scale_anchor_sizes(_LEGACY_CELL_DINO_ANCHOR_SIZES, target_size)
    elif anchor_profile == "cell_dino_tuned":
        sizes = _scale_anchor_sizes(_CELL_DINO_TUNED_ANCHOR_SIZES, target_size)
    else:
        raise ValueError(
            f"Unsupported anchor_profile '{anchor_profile}'. Expected one of: 'sam3_tuned', 'legacy'."
        )

    if anchor_aspect_ratios is not None:
        aspect_ratios = _coerce_anchor_aspect_ratios(anchor_aspect_ratios)
    elif anchor_profile == "sam3_tuned":
        aspect_ratios = _SAM3_TUNED_ASPECT_RATIOS
    elif anchor_profile == "cell_dino_tuned":
        aspect_ratios = _CELL_DINO_TUNED_ASPECT_RATIOS
    else:
        aspect_ratios = _LEGACY_CELL_DINO_ASPECT_RATIOS

    _validate_anchor_layout(sizes, aspect_ratios, num_levels)
    return AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)
