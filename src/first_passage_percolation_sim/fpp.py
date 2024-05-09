from __future__ import annotations

import copy
import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
import rustworkx as rw
from matplotlib import colors as mpl_colors
from rustworkx.generators import grid_graph

if TYPE_CHECKING:
    from numpy.random import Generator


def check_size_grid(size: int):
    if size % 2 == 0:
        msg = "The size of the grid side must be an odd number."
        raise ValueError(msg)


@dataclass
class CenteredGrid:
    """
    Represents a centered grid. That is, the center is at (0, 0) and
    the indices are from -size_side // 2 to size_side // 2 - 1.

    Attributes
    ----------
        array (np.ndarray): The 2-dimensional array representing the grid.

    Properties:
        size_side (int): The size of one side of the grid.
        start_index (int): The start index of the grid i.e. -size_side // 2.
        end_index (int): The end index of the grid i.e. size_side // 2 - 1.

    Methods
    -------
        __getitem__(key): Get the value at the specified key in the grid.
        __setitem__(key, value): Set the value at the specified key in the grid.

    Raises
    ------
        ValueError: If the array is not 2-dimensional.
        IndexError: If the index is out of bounds, i.e., less than -size_side // 2
            or greater than or equal to size_side // 2.

    """

    array: np.ndarray

    def __post_init__(self):
        if self.array.ndim != 2:  # noqa: PLR2004
            msg = "The array must be 2-dimensional."
            raise ValueError(msg)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.array})"

    def __str__(self):
        name = self.__class__.__name__
        indent = " " * len(name)
        array_str = np.array2string(
            self.array, separator=", ", prefix=indent, suffix=","
        )
        return f"{name}(\n{indent}{array_str}\n)"

    @property
    def size_side(self):
        return self.array.shape[0]

    @property
    def start_index(self):
        return -self._center

    @property
    def end_index(self):
        return self._center - 1

    @property
    def _center(self):
        return self.size_side // 2

    def _transform_index(self, key: int | slice):
        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            return slice(
                self._transform_index(start) if start is not None else None,
                self._transform_index(stop) if stop is not None else None,
                step,
            )

        key_arr = np.asarray(key)

        if (key_arr < -self._center).any() or (key_arr >= self._center).any():
            msg = f"Index {key} is out of bounds."
            raise IndexError(msg)
        return key + self._center

    def _transform_key(self, key: int | tuple) -> Any:
        if isinstance(key, tuple):
            return tuple(map(self._transform_index, key))
        return self._transform_index(key)

    def __getitem__(self, key):
        key = self._transform_key(key)
        return self.array[key]

    def __setitem__(self, key, value):
        key = self._transform_key(key)
        self.array[key] = value


class Dist(Protocol):
    def rvs(self, size: int, random_state: Generator) -> np.ndarray: ...


@functools.cache
def _get_lengths(graph: rw.PyGraph, source_node: int, dist: Dist, rng, nb_edges):
    x = dist.rvs(size=nb_edges, random_state=rng)
    i = 0

    def weight_func(_):
        nonlocal i
        w = x[i]
        i += 1
        return w

    return dict(rw.dijkstra_shortest_path_lengths(graph, source_node, weight_func))


class FirstPassagePercolation:
    """
    Represents a First Passage Percolation simulation.
        Each attribute can be modified after the object creation, except
        for the graph. However, some changes may require recomputing the lengths by
        calling the compute_lengths() method.

    Args:
    ----
            size_side (int): The size of the grid side.
            dist (Dist): The distribution object of the
                random edge weights.
            rng (numpy.random.Generator, optional): The random number generator.
                Defaults to np.random.default_rng().

    Attributes:
    ----------
            graph (Graph): The grid graph representing the simulation.
            size_side (int): The size of the grid side.
            dist (RandFunc): The random function used to generate edge weights.
            rng (numpy.random.Generator): The random number generator.
            center_node (int): The index of the center node.
            lengths (dict): The lengths of the shortest paths from the source node
                to all other nodes.
            grid_lengths (CenteredGrid): The lengths of the shortest paths from the
                source node in a grid format. The center node is at (0, 0). To get the
                center node to be at (size_side // 2, size_side // 2), take the raw
                array with grid_lengths.array.
            nb_nodes (int): The number of nodes in the graph.
            nb_edges (int): The number of edges in the graph.

    Methods:
    -------
            compute_lengths: Apply the Dijkstra algorithm to compute the lengths of the
                shortest paths from
                the source node to all other nodes.
            node_to_ij: Convert a node index to a (i, j) tuple.
            plot_heatmap: Plot a heatmap of the lengths.

    Examples:
    --------
    >>> import numpy as np
    >>> from scipy.stats import expon
    >>> from numpy.random import default_rng
    >>> from matplotlib import pyplot as plt
    >>> from first_passage_percolation.fpp import FirstPassagePercolation
    >>> size_side = 11
    >>> dist = expon(scale=1)
    >>> rng = default_rng(0)
    >>> fpp = FirstPassagePercolation(size_side, dist, rng).compute_lengths()
    >>> fpp.plot_heatmap()
    <matplotlib.image.AxesImage object at ...>

    We can keep the same object and change the distribution:
    >>> fpp.dist = expon(scale=2)
    >>> fpp.compute_lengths().plot_heatmap()
    <matplotlib.image.AxesImage object at ...>

    """

    def __init__(self, size_side, dist: Dist, rng=None):
        n = size_side
        self._size_side = size_side
        self._graph = grid_graph(n, n)
        self.dist = dist
        self.rng = rng or np.random.default_rng()
        self._lengths = None

        check_size_grid(size_side)
        if len(self.graph.edge_list()) != 2 * n * (n - 1):
            msg = "The number of edges is not correct."
            raise ValueError(msg)

    def __repr__(self):
        return (
            "FirstPassagePercolation("
            f"size_side={self.size_side}, "
            f"dist={self.dist}, "
            f"rng={self.rng}"
            ")"
        )

    def __eq__(self, value):
        if not isinstance(value, FirstPassagePercolation):
            return False
        return (
            self.size_side == value.size_side
            and self.dist == value.dist
            and self.rng == value.rng
        )

    def __hash__(self):
        return hash((self.size_side, self.dist, self.rng))

    def _clear_lengths(self):
        self._lengths = None
        self._grid_lengths = None

    def _set_lengths(self, lengths):
        self._lengths = lengths
        self._grid_lengths = CenteredGrid(self._get_grid_lengths_positive_indices())
        self._grid_lengths.array.flags.writeable = False  # make the array read-only

    @property
    def lengths(self):
        if self._lengths is None:
            msg = (
                "Lengths have not been set yet. "
                "Call the compute_lengths() method first."
            )
            raise ValueError(msg)
        return self._lengths

    @property
    def graph(self):
        return self._graph

    @property
    def size_side(self):
        return self._size_side

    @property
    def dist(self):
        return self._dist

    @dist.setter
    def dist(self, value: Dist):
        self._dist = value
        self._clear_lengths()

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, value):
        self._rng = value
        self._clear_lengths()

    @size_side.setter
    def size_side(self, value):
        self._size_side = value
        self._graph = grid_graph(value, value)

    @property
    def nb_nodes(self):
        return self.size_side**2

    @property
    def nb_edges(self):
        n = self.size_side
        return 2 * n * (n - 1)

    @property
    def center_node(self):
        n = self.size_side
        return (n - 1) // 2 * (n + 1)

    @property
    def grid_lengths(self):
        if self._grid_lengths is None:
            msg = (
                "Grid lengths have not been set yet. "
                "Call the compute_lengths() method first."
            )
            raise ValueError(msg)
        return self._grid_lengths

    def compute_lengths(self):
        """
        Apply the Dijkstra algorithm to compute the lengths of the shortest
        paths from the source node to all other nodes.

        Returns
        -------
            FirstPassagePercolation: The object itself.

        """
        # initialize the random number generator, so that we can reproduce the results
        rng = copy.deepcopy(self.rng)

        # _lengths is a dict node_id(int): length
        lengths = _get_lengths(
            self.graph, self.center_node, self.dist, rng, self.nb_edges
        )
        # 0 is omitted in the dict, so we add it manually
        lengths[self.center_node] = 0
        self._set_lengths(lengths)
        return self

    def node_to_ij(self, node):
        """
        Convert a node index to a (i, j) tuple.

        Args:
        ----
            node (int | np.ndarray): The node index or indices.

        Returns:
        -------
            tuple: The (i, j) tuple. If node is an array, the output is
                (array of i, array of j)

        """
        n = self.size_side
        row, col = np.divmod(node, n)
        return row - n // 2, col - n // 2

    def _get_grid_lengths_positive_indices(self):
        """Return a size_side x size_side array of the lengths from the source node."""
        grid = np.zeros(self.nb_nodes, dtype=float)
        lengths = self.lengths
        indices = np.fromiter(lengths.keys(), dtype=int, count=self.nb_nodes - 1)
        vals = np.fromiter(lengths.values(), dtype=float, count=self.nb_nodes - 1)
        grid[indices] = vals
        return grid.reshape(self.size_side, self.size_side)

    def plot_heatmap(self, ax=None, **kwargs):
        """
        Plot a heatmap of the lengths.

        Args:
        ----
            ax (Axes | None): The matplotlib axes.
            **kwargs: Additional arguments passed to plt.imshow
                (cmap, interpolation, etc.).

        """
        ax = ax or plt.gca()
        return ax.imshow(self.grid_lengths.array, **kwargs)

    def plot_progression(self, t, ax=None, colors=("red", "white"), **kwargs):
        """
        Draw the set T(t) := {i : L(i) <= t} in the first color
        and its complement in the second color.

        Args:
        ----
            t (float): The threshold.
            ax (Axes | None): The matplotlib axes.
            colors (tuple[str, str]): The colors of the two sets.
            **kwargs: Additional arguments passed to plt.plot.

        """
        ax = ax or plt.gca()

        grid = self.grid_lengths.array
        cmap = mpl_colors.ListedColormap(tuple(reversed(colors)))
        ax.imshow(grid <= t, cmap=cmap, **kwargs)
        return ax


def lengths_varying_param(
    dist_func, range_x, size_side, rng, *, positive_indices=False
):
    """
    Generate the grid lengths with varying parameters of the distribution that
    describes the random edge weights.

    Args:
    ----
        dist_func: A function that takes a parameter and return the distribution of
            the random edge weights.
        range_x: An iterable representing the range of parameter values.
        size_side: The size of the grid side.
        rng: The random number generator.
        positive_indices (bool): If True, the every index is positive and the center
            node is at (size_side // 2, size_side // 2). Otherwise, the center node
            is at (0, 0). Defaults to False.

    Yields:
    ------
        The grid lengths with positive indices for each parameter value.

    """
    it = iter(range_x)
    ffp = FirstPassagePercolation(size_side, dist_func(next(it)), rng).compute_lengths()

    def get_lengths(ffp):
        if positive_indices:
            return ffp.grid_lengths.array
        return ffp.grid_lengths

    yield get_lengths(ffp)
    for i in it:
        ffp.dist = dist_func(i)
        ffp.compute_lengths()
        yield get_lengths(ffp)


def plot_lengths_varying_param(
    dist_func, range_x, size_side, rng, *, nb_cols=1, name_x="Parameter", **kwargs
):
    """
    Plot the grid lengths with varying parameters.

    Args:
    ----
        dist_func: A function that takes a parameter and return the distribution of
            the random edge weights.
        range_x: An iterable representing the range of parameter values.
        size_side: The size of the grid side.
        rng: The random number generator.
        nb_cols: The number of columns in the plot.
        name_x: The name of the parameter (appear in the title of each subplot)
        **kwargs: Additional arguments passed to plt.imshow (cmap, interpolation, etc.).

    """
    nb_vars = len(range_x)
    nb_rows = nb_vars // nb_cols + (nb_vars % nb_cols != 0)
    fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(5 * nb_cols, 5 * nb_rows))
    all_mat = lengths_varying_param(
        dist_func, range_x, size_side, rng, positive_indices=True
    )
    for ax, grid_lengths, x in zip(axes.ravel(), all_mat, range_x, strict=False):
        ax.imshow(grid_lengths, **kwargs)
        ax.set_title(f"{name_x} = {x:.2f}")
    return fig, axes
