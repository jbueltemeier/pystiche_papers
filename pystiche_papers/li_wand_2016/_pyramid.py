from typing import Any, Optional, Sequence, Union

from pystiche import pyramid

__all__ = ["image_pyramid"]


def image_pyramid(
    impl_params: bool = True,
    max_edge_size: int = 384,
    num_steps: Optional[Union[int, Sequence[int]]] = None,
    num_levels: Optional[int] = None,
    min_edge_size: int = 64,
    edge: Union[str, Sequence[str]] = "long",
    **octave_image_pyramid_kwargs: Any,
) -> pyramid.OctaveImagePyramid:
    if num_steps is None:
        # https://github.com/pmeier/CNNMRF/blob/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f/cnnmrf.lua#L44
        num_steps = 100 if impl_params else 200

    if num_levels is None:
        # https://github.com/pmeier/CNNMRF/blob/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f/cnnmrf.lua#L43
        num_levels = 3 if impl_params else None

    return pyramid.OctaveImagePyramid(
        max_edge_size,
        num_steps,
        num_levels=num_levels,
        min_edge_size=min_edge_size,
        edge=edge,
        **octave_image_pyramid_kwargs,
    )
