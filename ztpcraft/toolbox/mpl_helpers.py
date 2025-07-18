import numpy as np
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType
from typing import Any, Dict, Union, Literal, List, Tuple
from ztpcraft.utils.figure_settings import MPL_PRESET
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def update_matplotlib_settings(
    settings: Union[
        Dict[str, Any], Literal["PhysRevOneCol", "PhysRevOneColSans", "PhysRevTwoCol"]
    ],
):
    """
    Update the matplotlib settings.

    Parameters
    ----------
    settings : Union[Dict[str, Any]]
        The settings to update the matplotlib. If a string is given, it should
        be a key in the MPL_PRESET dictionary.
    """
    if isinstance(settings, str):
        try:
            settings = MPL_PRESET[settings]
        except KeyError:
            raise KeyError(f"settings {settings} not found in MPL_PRESET.")
    plt.rcParams.update(settings)


def preview_palette(color_palette: List) -> Tuple[Figure, Axes]:
    """
    Plot the color palette for a list of color palettes.
    """
    fig, ax = plt.subplots()
    # plot the color palette as boxes of color with height of 1 and width of 1
    # using bar chart
    for i, color in enumerate(color_palette):
        ax.bar(i, 1, color=color)
    return fig, ax


def single_hex_to_rgb(hc: str) -> Tuple[float, float, float]:
    hc = hc.lstrip("#")
    if len(hc) != 6:
        raise ValueError(f"Input '{hc}' is not in the correct format '#RRGGBB'.")
    try:
        r = int(hc[0:2], 16) / 255.0
        g = int(hc[2:4], 16) / 255.0
        b = int(hc[4:6], 16) / 255.0
    except ValueError:
        raise ValueError(f"Input '{hc}' contains invalid hexadecimal digits.")
    return (r, g, b)


def hex_to_rgb_normalized(
    hex_color: Union[str, List[str], np.ndarray],
) -> Union[Tuple[float, float, float], np.ndarray]:
    """
    Convert hex color(s) to normalized RGB.

    Parameters:
        hex_color (str or list/array of str): Hex color string(s) (e.g., "#341D5F" or ["#341D5F", "#FFAABB"]).

    Returns:
        tuple or np.ndarray: Normalized RGB values.
            - Single input: Tuple of (R, G, B).
            - Multiple inputs: 2D NumPy array with shape (n, 3).
    """

    if isinstance(hex_color, str):
        return single_hex_to_rgb(hex_color)
    elif isinstance(hex_color, (list, np.ndarray)):
        rgb_list = [single_hex_to_rgb(hc) for hc in hex_color]
        return np.array(rgb_list)
    else:
        raise TypeError("hex_color must be a string or a list/array of strings.")


def single_rgb_to_hex(rgb_tuple: Tuple[float, float, float]) -> str:
    if any(not (0.0 <= component <= 1.0) for component in rgb_tuple):
        raise ValueError("All RGB components must be in the range [0, 1].")
    r, g, b = rgb_tuple
    return "#{:02X}{:02X}{:02X}".format(
        int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
    )


def rgb_normalized_to_hex(
    rgb: Union[
        Tuple[float, float, float], List[Tuple[float, float, float]], np.ndarray
    ],
) -> Union[str, List[str], np.ndarray]:
    """
    Convert normalized RGB to hex color string(s).

    Parameters:
        rgb (tuple or list/array of tuples): Normalized RGB values.
            - Single input: Tuple of (R, G, B).
            - Multiple inputs: List or 2D NumPy array with shape (n, 3).

    Returns:
        str or list of str or np.ndarray: Hex color string(s).
            - Single input: Hex string (e.g., "#341D5F").
            - Multiple inputs: List or array of hex strings.
    """

    if isinstance(rgb, tuple):
        return single_rgb_to_hex(rgb)
    elif isinstance(rgb, (list, np.ndarray)):
        return np.array([single_rgb_to_hex(tuple(color)) for color in rgb])
    else:
        raise TypeError("rgb must be a tuple or a list/array of tuples.")


# def hex_to_rgb_normalized(hex_color) -> np.ndarray:
#     """
#     Convert a hex color string to normalized RGB tuple.

#     Parameters:
#         hex_color (str): Hex color string (e.g., "#341D5F" or "341D5F").

#     Returns:
#         tuple: A tuple containing normalized RGB values as floats (R, G, B) in [0, 1].
#     """
#     hex_color = hex_color.lstrip("#")

#     if len(hex_color) != 6:
#         raise ValueError(f"Input '{hex_color}' is not in the correct format '#RRGGBB'.")

#     try:
#         r = int(hex_color[0:2], 16) / 255.0
#         g = int(hex_color[2:4], 16) / 255.0
#         b = int(hex_color[4:6], 16) / 255.0
#     except ValueError:
#         raise ValueError(f"Input '{hex_color}' contains invalid hexadecimal digits.")

#     return r, g, b


def hex_to_hsv(hex_color: str) -> np.ndarray:
    """
    Convert a hex color to HSV.

    Parameters
    ----------
    hex_color : str
        The hex color.

    Returns
    -------
    Tuple[float, float, float]
        The HSV values.
    """
    return rgb_to_hsv(hex_to_rgb_normalized(hex_color))


def hsv_to_hex(hsv_color: np.ndarray) -> str:
    """
    Convert an HSV color to hex.

    Parameters
    ----------
    hsv_color : np.ndarray
        The HSV color.

    Returns
    -------
    str
        The hex color.
    """
    return rgb_normalized_to_hex(hsv_to_rgb(hsv_color))


def add_zoom_inset(
    ax: Axes,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    position: Tuple[float, float] = (0.6, 0.6),
    size: Tuple[float, float] = (0.3, 0.3),
    anchor: str = "lower left",
    mark: bool = True,
    loc1: int = 2,
    loc2: int = 4,
    linestyle: str = "-",
    mark_ec: ColorType = "0.5",
):
    """
    Adds a zoomed-in inset to a matplotlib Axes.

    Parameters:
    - ax: parent matplotlib Axes
    - xlim, ylim: data limits of the zoomed inset region
    - position: (x,y) tuple in axes coordinates for pinning inset position
    - size: (width, height) of inset relative to parent axes content area
    - anchor: inset corner to anchor to position ('lower left', 'upper right', etc.)
    """

    fig = ax.figure

    # Determine inset position based on anchor
    anchor_dict = {
        "lower left": (0, 0),
        "lower right": (-1, 0),
        "upper left": (0, -1),
        "upper right": (-1, -1),
        "center": (-0.5, -0.5),
    }

    # Get actual content region (excluding ticks, labels)
    renderer = fig.canvas.get_renderer()
    bbox_content = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())

    # Convert position & size from axes coords to figure coords, accurately
    x0 = bbox_content.x0 + position[0] * bbox_content.width
    y0 = bbox_content.y0 + position[1] * bbox_content.height
    width = size[0] * bbox_content.width
    height = size[1] * bbox_content.height

    offset_x, offset_y = anchor_dict[anchor]
    fig_left = x0 + offset_x * width
    fig_bottom = y0 + offset_y * height

    # Create inset Axes
    axins = fig.add_axes([fig_left, fig_bottom, width, height])
    axins.set_xlim(*xlim)
    axins.set_ylim(*ylim)
    axins.set_xticks([])
    axins.set_yticks([])

    # Copy plot lines
    for line in ax.lines:
        axins.plot(
            line.get_xdata(),
            line.get_ydata(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            marker=line.get_marker(),
        )

    # Copy scatter plots
    for coll in ax.collections:
        if isinstance(coll, PathCollection):
            offsets = coll.get_offsets()
            axins.scatter(
                offsets[:, 0],
                offsets[:, 1],
                c=coll.get_facecolors(),
                s=coll.get_sizes(),
                marker=coll.get_paths()[0],
                linewidths=coll.get_linewidths(),
                edgecolors=coll.get_edgecolors(),
                alpha=coll.get_alpha(),
                cmap=coll.get_cmap(),
                norm=coll.norm,
            )
        elif isinstance(coll, QuadMesh):  # pcolormesh
            coords = coll.get_coordinates()
            arr = coll.get_array().reshape(coords.shape[0] - 1, coords.shape[1] - 1)
            axins.pcolormesh(
                coords[..., 0],
                coords[..., 1],
                arr,
                cmap=coll.get_cmap(),
                shading=coll.get_shading(),
                norm=coll.norm,
                alpha=coll.get_alpha(),
            )

    # Copy images (imshow)
    for im in ax.images:
        if isinstance(im, AxesImage):
            axins.imshow(
                im.get_array(),
                extent=im.get_extent(),
                origin=im.origin,
                cmap=im.get_cmap(),
                interpolation=im.get_interpolation(),
                norm=im.norm,
                alpha=im.get_alpha(),
            )
    axins.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) * height / width))
    if mark:
        mark_inset(
            ax, axins, loc1=loc1, loc2=loc2, fc="none", ec=mark_ec, linestyle=linestyle
        )
    return axins
