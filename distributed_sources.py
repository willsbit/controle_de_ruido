import numpy as np
import polars as pl
import matplotlib.pyplot as plt

c0 = 343  # [m/s]
rho0 = 1.2  # [kg/m^3]
p_ref = 2e-5  # [Pa]
a = 28e-3 / 2  # [m]
r = 1  # [m]

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
    .with_columns(pl.col("x", "y", "z") / 10)
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

field_points = pl.DataFrame({"x": [0], "y": [0], "z": [1]}).with_row_index(name="field_point_id")

elements_with_euclidean_distances = elements_with_coordinates.join(field_points, how="cross").select(
    pl.col("field_point_id", "area"),
    (
        np.sqrt(
            (pl.col("x") - pl.col("x_bar")) ** 2
            + (pl.col("y") - pl.col("y_bar")) ** 2
            + (pl.col("z") - pl.col("z_bar")) ** 2
        )
    ).alias("distance"),
)

velocities_per_frequency = velocity.with_columns(
    (pl.col("frequency") * 2 * np.pi).alias("omega"), ((pl.col("frequency") * 2 * np.pi) / c0).alias("k")
).sort(by="frequency")

rayleigh_integral_per_frequency = (
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

pressure_per_frequency = rayleigh_integral_per_frequency.select(
    pl.col("frequency"),
    (10 * ((0.5 * (pl.col("re").pow(2) + pl.col("im").pow(2))) / (p_ref**2)).log(base=10)).alias("Lp"),
).sort(by="frequency")
pressure_per_frequency

# theoretical_model
velocities_array = velocities_per_frequency.sort(by="frequency").to_numpy(structured=True)

velocities_array
far_field_pressure = (
    -1j
    * 0.5
    * rho0
    * c0
    * velocities_array["u"]
    * velocities_array["k"]
    * a
    * (a / r)
    * np.exp(1j * velocities_array["k"] * r)
)
far_field_Lp = 10 * np.log10((0.5 * np.real(far_field_pressure * np.conj(far_field_pressure))) / (p_ref**2))

fig, ax = plt.subplots()
ax.semilogx(velocities_array["frequency"], far_field_Lp, label="Apoximação campo distante", linestyle="--")
ax.semilogx(
    pressure_per_frequency.select("frequency"), pressure_per_frequency.select("Lp"), label="Simulado", alpha=0.5
)
ax.legend()
