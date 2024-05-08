from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Self, overload

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy import floating, random, signedinteger
from numpy.typing import NBitBase, NDArray
from rustworkx import PyGraph

type FloatArray = NDArray[floating[Any]]

def check_size_grid(size: int) -> None: ...
@dataclass
class CenteredGrid:
    array: FloatArray
    def __post_init__(self) -> None: ...
    @property
    def size_side(self) -> int: ...
    @property
    def start_index(self) -> int: ...
    @property
    def end_index(self) -> int: ...
    def __getitem__(self, key) -> FloatArray | float: ...
    def __setitem__(self, key, value) -> None: ...

class Dist(Protocol):
    def rvs(self, size: int, random_state: random.Generator) -> NDArray: ...

class FirstPassagePercolation:
    def __init__(
        self, size_side: int, dist: Dist, rng: random.Generator | None = None
    ) -> None: ...
    def __eq__(self, value: object) -> bool: ...
    def __hash__(self) -> int: ...
    @property
    def lengths(self): ...
    @property
    def graph(self) -> PyGraph[int, float]: ...
    @property
    def size_side(self) -> int: ...
    @property
    def dist(self) -> Dist: ...
    @dist.setter
    def dist(self, value: Dist) -> None: ...
    @property
    def rng(self) -> random.Generator: ...
    @rng.setter
    def rng(self, value: random.Generator) -> None: ...
    @size_side.setter
    def size_side(self, value: int) -> None: ...
    @property
    def nb_nodes(self) -> int: ...
    @property
    def nb_edges(self) -> int: ...
    @property
    def center_node(self) -> int: ...
    @property
    def grid_lengths(self) -> CenteredGrid: ...
    def compute_lengths(self) -> Self: ...
    def node_to_ij(
        self, node: int | NDArray
    ) -> tuple[
        int | NDArray[signedinteger[Any]], int | NDArray[signedinteger[Any]]
    ]: ...
    def plot_heatmap(self, ax: Axes | None = None, **kwargs) -> AxesImage: ...
    def plot_progression(
        self, t: float, ax: Axes | None = None, colors: tuple[str, str] = ..., **kwargs
    ) -> Axes: ...

@overload
def lengths_varying_param[T](
    dist_func: Callable[[T], Dist],
    range_x: Iterable[T],
    size_side: int,
    rng: random.Generator,
    *,
    positive_indices: Literal[True],
) -> Generator[NDArray, Any, None]: ...
@overload
def lengths_varying_param[T](
    dist_func: Callable[[T], Dist],
    range_x: Iterable[T],
    size_side: int,
    rng: random.Generator,
    *,
    positive_indices: Literal[False] = False,
) -> Generator[CenteredGrid, Any, None]: ...
def lengths_varying_param[T](
    dist_func: Callable[[T], Dist],
    range_x: Iterable[T],
    size_side: int,
    rng: random.Generator,
    *,
    positive_indices: bool = False,
) -> Generator[NDArray | CenteredGrid, Any, None]: ...
def plot_lengths_varying_param[T](
    dist_func: Callable[[T], Dist],
    range_x: Iterable[T],
    size_side: int,
    rng: random.Generator,
    nb_cols=1,
    name_x: str = ...,
    **kwargs,
) -> tuple[Figure, Axes]: ...
