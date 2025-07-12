import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from tmm import _utils as utils
from numpy.typing import NDArray, ArrayLike
from scipy.special import j1


c0 = 343  # [m/s]
rho0 = 1.2  # [kg/m^3]
p_ref = 2e-5  # [Pa]
a = 28e-2 / 2  # [m]
space_discretization = 0.028  # [m]


def get_distances_to_field_points(elements_with_coordinates: pl.DataFrame, field_points: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the euclidean distances between the center of each mesh element and the field points,
    the outer points for which we want to calculate the pressure values.

    Args:
        elements_with_coordinates: DataFrame containing the coordinates of the mesh elements.
        field_points: DataFrame containing the coordinates of the field points.

    Returns:
        DataFrame containing the distances between the center of each mesh element and the field points.
    """
    return elements_with_coordinates.join(field_points, how="cross").select(
        pl.col("field_point_id", "area"),
        (
            np.sqrt(
                (pl.col("x") - pl.col("x_bar")) ** 2
                + (pl.col("y") - pl.col("y_bar")) ** 2
                + (pl.col("z") - pl.col("z_bar")) ** 2
            )
        ).alias("distance"),
    )


def calculate_pressure_per_frequency(
    elements_with_euclidean_distances: pl.DataFrame, velocities_per_frequency: pl.DataFrame
) -> pl.DataFrame:
    """
    Calculate the real and imaginary parts of the pressure, per frequency for each element.
    This is done by calculating the Rayleigh integral summing the contributions of each monopole in the mesh centroids.

    Args:
        elements_with_euclidean_distances: DataFrame containing elements with their Euclidean distances. Should also contain the area for each mesh element.
        velocities_per_frequency: DataFrame containing volume velocities of the piston per frequency.

    Returns:
        DataFrame containing the real and imaginary parts of the pressure, per frequency for each element.
    """
    return (
        elements_with_euclidean_distances.join(velocities_per_frequency, how="cross")
        .with_columns(((pl.col("omega") * pl.col("u") * pl.col("area")) / pl.col("distance")).alias("mul_factors"))
        .select(
            pl.col("field_point_id", "frequency"),
            ((pl.col("k") * pl.col("distance")).cos() * pl.col("mul_factors")).alias("re"),
            ((pl.col("k") * pl.col("distance")).sin() * pl.col("mul_factors")).alias("im"),
        )
        .group_by("field_point_id", "frequency")
        .sum()
        .with_columns(
            # multiply by -j*rho0 / 2pi
            (pl.col("re") * (-rho0 / (2 * np.pi))).alias("im"),
            (pl.col("im") * (rho0 / (2 * np.pi))).alias("re"),
        )
    )


def calculate_lp_per_frequency(pressure_per_frequency: pl.DataFrame) -> pl.DataFrame:
    """
    Transforms the real and imaginary parts of the pressure into a logarithmic pressure level.

    Args:
        pressure_per_frequency: DataFrame containing the pressure data.
    Returns:
        DataFrame containing the logarithmic pressure level.
    """
    return pressure_per_frequency.select(
        pl.col("field_point_id", "frequency"),
        (10 * ((0.5 * (pl.col("re").pow(2) + pl.col("im").pow(2))) / (p_ref**2)).log(base=10)).alias("Lp"),
    ).sort(by="frequency")


def analytic_pressure_per_frequency(
    velocities_array, r: NDArray[np.float64] | np.float64, theta: NDArray[np.float64] | np.float64
) -> NDArray[np.float64]:
    """
    Calculates pressure per point, per frequency using a far-field approximation.

    Args:
        velocities_array: DataFrame containing the velocity data.
        r: radius, in meters.
        theta: angle, in radians.

    Returns:
        Array containing the analytic pressure per frequency.
    """
    bessel_argument = velocities_array["k"] * a * np.sin(theta)
    non_zero_mask = bessel_argument != 0
    bessel_term = np.ones_like(bessel_argument)
    bessel_term[non_zero_mask] = 2 * (j1(bessel_argument[non_zero_mask])) / (bessel_argument[non_zero_mask])

    return (
        -1j
        * 0.5
        * rho0
        * c0
        * velocities_array["u"]
        * velocities_array["k"]
        * a
        * (a / r)
        * np.exp(1j * velocities_array["k"] * r)
        * bessel_term
    )


def analytic_lp_per_frequency(analytic_pressure_per_frequency: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculate the analytic pressure level, per point, per frequency.

    Args:
        analytic_pressure_per_frequency: Array containing the analytic pressure per frequency for each field point.

    Returns:
        Array containing the analytic pressure level per frequency for each field point.
    """
    return 10 * np.log10(
        (0.5 * np.real(analytic_pressure_per_frequency * np.conj(analytic_pressure_per_frequency))) / (p_ref**2)
    )


def kinsler_pressure_per_frequency(velocities_array: NDArray[np.float64], r: float) -> NDArray[np.float64]:
    """
    Calculates pressure per point, per frequency using on the z-axis
    using the expression seen on Kinsler's Fundamentals of Acoustics, chapter 7.

    Args:
        velocities_array: DataFrame containing the velocity data.
        r: radius, in meters.

    Returns:
        Array containing the analytic pressure per frequency.
    """
    decay_factor = 1 - np.exp(1j * velocities_array["k"] * (np.sqrt(r**2 + a**2) - r))
    return rho0 * c0 * velocities_array["u"] * decay_factor * np.exp(1j * velocities_array["k"] * r)


def create_arc_points(center_x, center_y, radius, start_angle_deg, end_angle_deg, spacing):
    """
    Creates points along an arc defined by a center point, radius, and angle range.

    Args:
        center_x: x-coordinate of the center point.
        center_y: y-coordinate of the center point.
        radius: radius of the arc.
        start_angle_deg: starting angle of the arc in degrees.
        end_angle_deg: ending angle of the arc in degrees.
        spacing: angular spacing between points in degrees.

    Returns:
        Tuple containing arrays of angles in radians, z-coordinates, and y-coordinates.
    """
    angles = np.arange(start_angle_deg, end_angle_deg + spacing, spacing)
    angles_rad = np.deg2rad(angles)
    y_coords = center_y + radius * np.sin(angles_rad)
    z_coords = center_x + radius * np.cos(angles_rad)
    return angles_rad, z_coords, y_coords


# %% import data
elements_schema = {
    "node1": pl.UInt32,
    "node2": pl.UInt32,
    "node3": pl.UInt32,
}
elements = pl.read_csv(
    "distributed_source_data/elements.txt",
    separator=",",
    has_header=False,
    new_columns=["node1", "node2", "node3"],
).cast(elements_schema)
nodes = (
    pl.read_csv(
        "distributed_source_data/nodes.txt",
        separator=",",
        has_header=False,
        new_columns=["x", "y", "z"],
    )
    .with_row_index(name="node", offset=1)
    .with_columns(pl.col("x", "y", "z"))
)

velocity = pl.read_csv(
    "distributed_source_data/velocidade.txt",
    separator=",",
    has_header=False,
    new_columns=["frequency", "u"],
)

elements_with_coordinates = (
    elements.join(
        nodes.select("node", pl.col("x", "y", "z").name.suffix("_node1")),
        left_on="node1",
        right_on="node",
    )
    .join(
        nodes.select("node", pl.col("x", "y", "z").name.suffix("_node2")),
        left_on="node2",
        right_on="node",
    )
    .join(
        nodes.select("node", pl.col("x", "y", "z").name.suffix("_node3")),
        left_on="node3",
        right_on="node",
    )
    .with_columns(
        pl.mean_horizontal("x_node1", "x_node2", "x_node3").alias("x_bar"),
        pl.mean_horizontal("y_node1", "y_node2", "y_node3").alias("y_bar"),
        pl.mean_horizontal("z_node1", "z_node2", "z_node3").alias("z_bar"),
        (
            np.abs(
                pl.col("x_node1") * (pl.col("y_node2") - pl.col("y_node3"))
                + pl.col("x_node2") * (pl.col("y_node3") - pl.col("y_node1"))
                + pl.col("x_node3") * (pl.col("y_node1") - pl.col("y_node2"))
            )
            * 0.5
        ).alias("area"),
    )
)

velocities_per_frequency = velocity.with_columns(
    (pl.col("frequency") * 2 * np.pi).alias("omega"), ((pl.col("frequency") * 2 * np.pi) / c0).alias("k")
).sort(by="frequency")


# %% a)
r = 1  # [m]
field_points = pl.DataFrame({"x": [0], "y": [0], "z": [1]}).with_row_index(name="field_point_id")

elements_with_euclidean_distances = get_distances_to_field_points(elements_with_coordinates, field_points)
pressure_per_frequency = calculate_pressure_per_frequency(elements_with_euclidean_distances, velocities_per_frequency)
lp_per_frequency = calculate_lp_per_frequency(pressure_per_frequency)

# theoretical_model
velocities_array = velocities_per_frequency.sort(by="frequency").to_numpy(structured=True)
analytic_pressure = analytic_pressure_per_frequency(velocities_array, r, 0)
analytic_Lp = analytic_lp_per_frequency(analytic_pressure)

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(velocities_array["frequency"], analytic_Lp, label="Campo distante", linestyle="--", linewidth=2)
ax.semilogx(lp_per_frequency.select("frequency"), lp_per_frequency.select("Lp"), label="Simulado", alpha=0.8)

center_frequencies = utils.nth_octave(1, fmin=10, fmax=5000)[:, 1]
ax.set_xlabel("Frequência [Hz]")
ax.set_ylabel(r"$L_p$ [dB]")
ax.set_xticks(center_frequencies)
ax.set_xticks([], minor=True)
ax.set_xticklabels([str(int(f)) if f.is_integer() else str(f) for f in center_frequencies])
ax.grid()
ax.set_xlim(10, 5000)
ax.set_ylim(0, 120)
ax.legend(loc="upper left")
plt.savefig("analitico_vs_simulado_1m.svg", format="svg", bbox_inches="tight")


# %% b)
r = np.arange(start=0.01, stop=0.4 + 0.028, step=0.028)
velocity_2khz = velocities_per_frequency.filter(pl.col("frequency") == 2001)
velocity_2khz_array = velocity_2khz.sort(by="frequency").to_numpy(structured=True)

field_points = (
    pl.DataFrame(r, schema=["z"]).with_columns(x=pl.lit(0.0), y=pl.lit(0.0)).with_row_index(name="field_point_id")
)
elements_with_euclidean_distances = get_distances_to_field_points(elements_with_coordinates, field_points)
pressure_per_frequency = calculate_pressure_per_frequency(elements_with_euclidean_distances, velocity_2khz).join(
    field_points, on="field_point_id"
)

analytic_pressure = analytic_pressure_per_frequency(velocity_2khz_array, r, 0)
kinsler_analytic_pressure = kinsler_pressure_per_frequency(velocity_2khz_array, r)

categories = [r"$\mathfrak{Re}(p)$", r"$\mathfrak{Im}(p)$"]
fig, ax = plt.subplots(figsize=(8, 5))
(p1,) = ax.plot(
    r,
    np.real(kinsler_analytic_pressure),
    label=r"Analítico, $\mathfrak{Re}(p)$",
    linestyle="--",
    linewidth=2,
    color="black",
)

(p1_farfield,) = ax.plot(
    r,
    np.real(analytic_pressure),
    label=r"Analítico - campo distante, $\mathfrak{Re}(p)$",
    linestyle="-.",
    linewidth=1,
    color="black",
    alpha=0.4,
)

(p2,) = ax.plot(
    pressure_per_frequency.select("z"),
    pressure_per_frequency.select("re"),
    label=r"Simulado, $\mathfrak{Re}(p)$",
    alpha=0.8,
)
(p3,) = ax.plot(
    r,
    np.imag(kinsler_analytic_pressure),
    label=r"Analítico, $\mathfrak{Im}(p)$",
    linestyle="--",
    linewidth=2,
    color="darkgreen",
)

(p3_farfield,) = ax.plot(
    r,
    np.imag(analytic_pressure),
    label=r"Analítico - campo distante, $\mathfrak{Im}(p)$",
    linestyle="-.",
    linewidth=1,
    color="darkgreen",
    alpha=0.4,
)


(p4,) = ax.plot(
    pressure_per_frequency.select("z"),
    pressure_per_frequency.select("im"),
    label=r"Simulado, $\mathfrak{Im}(p)$",
    alpha=0.8,
)
(p5,) = ax.plot([0], marker="None", linestyle="None", label="dummy-tophead")

ax.set_xlabel(r"Coordenada $z$ [m]")
ax.set_ylabel(r"$p$ [Pa]")
ax.grid()
ax.set_xlim(0, 0.4)
ax.set_ylim(-40, 40)

ax.legend(
    [p5, p1_farfield, p3_farfield, p5, p1, p3, p5, p2, p4],
    [r"Campo distante"] + categories + [r"Analítico"] + categories + ["Simulado"] + categories,
    loc="lower right",
    ncol=3,
)

plt.savefig("analitico_vs_simulado_eixo_z_2khz.svg", format="svg", bbox_inches="tight")


# %% c)
velocity_2khz = velocities_per_frequency.filter(pl.col("frequency") == 2001)
velocity_2khz_array = velocity_2khz.sort(by="frequency").to_numpy(structured=True)
theta, z, y = create_arc_points(center_x=0, center_y=0, radius=1, start_angle_deg=90, end_angle_deg=-90, spacing=-5)

field_points = (
    pl.DataFrame([y, z, theta], schema=["y", "z", "theta"])
    .with_columns(x=pl.lit(0.0))
    .with_row_index(name="field_point_id")
)
elements_with_euclidean_distances = get_distances_to_field_points(elements_with_coordinates, field_points)
pressure_per_frequency = calculate_pressure_per_frequency(elements_with_euclidean_distances, velocity_2khz).join(
    field_points, on="field_point_id"
)

directivity_at_point = pressure_per_frequency.select(
    pl.col("field_point_id", "y", "z", "theta"),
    (pl.col("re").pow(2) + pl.col("im").pow(2)).alias("abs_pressure"),
).with_columns((10 * (pl.col("abs_pressure") / pl.max("abs_pressure")).log(base=10)).alias("directivity"))

analytic_pressure = analytic_pressure_per_frequency(velocity_2khz_array, r=1, theta=theta)
analytic_abs_pressure_sq = np.real(analytic_pressure * np.conj(analytic_pressure))

analytic_directivity = 10 * np.log10(analytic_abs_pressure_sq / np.max(analytic_abs_pressure_sq))

fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
ax.plot(
    directivity_at_point.select("theta"),
    directivity_at_point.select("directivity"),
    label="Simulado",
    color="orange",
    linewidth=2,
)
ax.plot(theta, analytic_directivity, label="Campo distante", linestyle="--", alpha=0.8)
ax.set_theta_zero_location("N")  # 0 degrees at the top
ax.set_theta_direction(-1)  # clockwise
ax.legend(loc="best")
ax.set_rticks([0, -10, -20, -30])
ax.set_thetalim(np.deg2rad([-90, 90]))
ax.set_xlabel("Diretividade [dB]")
ax.xaxis.set_label_coords(0.5, 0.15)
plt.savefig("diretividade_2khz.svg", format="svg", bbox_inches="tight")
plt.show()

# %% d)
velocity_3khz = velocities_per_frequency.filter(pl.col("frequency") == 3001)
velocity_3khz_array = velocity_3khz.sort(by="frequency").to_numpy(structured=True)

x = np.arange(-0.4, 0.4 + space_discretization, space_discretization)
z = np.arange(0.01, 0.4 + space_discretization, space_discretization)
xx, zz = np.meshgrid(x, z)

grid = (
    pl.DataFrame([xx.ravel(), zz.ravel()], schema=["x", "z"])
    .with_columns(y=pl.lit(0.0))
    .with_row_index(name="field_point_id")
)

elements_with_euclidean_distances = get_distances_to_field_points(elements_with_coordinates, grid)
pressure_per_frequency = calculate_pressure_per_frequency(elements_with_euclidean_distances, velocity_3khz).join(
    grid, on="field_point_id"
)

# pivot dataframe into the expected format by plt.contourf
contour_pressures = pressure_per_frequency.pivot(values="re", index="z", on="x").drop("z").to_numpy()

# adjust scales
max_abs_val = np.nanmax(np.abs(contour_pressures))
vmin, vmax = -max_abs_val, max_abs_val

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
contour = plt.contourf(
    zz, xx, contour_pressures, levels=1000, vmin=vmin, vmax=vmax, cmap="seismic", norm="linear", extend="neither"
)
contour.set_edgecolor("face")

plt.colorbar(format="%.1f", label=r"$\mathfrak{Re}(p)$ [Pa]", spacing="uniform", fraction=0.028, pad=0.01)
ax.set_xlabel("z [m]", labelpad=0)
ax.set_ylabel("x [m]", labelpad=5)
ax.set_aspect("equal", adjustable="box")
plt.savefig("contourplot_xz_3khz.svg", format="svg", bbox_inches="tight")
plt.show()
