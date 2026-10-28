# ============================================================
# CELL 1
# IMPORT PACKAGES
# ============================================================

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import trimesh

from Bio.PDB import PDBParser

import tidy3d as td
import tidy3d.web as web

# ============================================================
# CELL 2
# USER SETTINGS
# ============================================================

PDB_FILE = "1mbn.pdb"

DATA_DIR = "./data"
Path(DATA_DIR).mkdir(
    parents=True,
    exist_ok=True,
)

# Geometry
SPHERE_RADIUS_A = 2.0
ANGSTROM_TO_UM = 1e-4

# Materials
EPS_WATER = 1.33**2
EPS_PROTEIN = 1.6**2

# External electric field
E_EXT = 74.0

# Vacuum permittivity
eps0 = 8.8541878128e-12

# Target wavelength: 670 nm
WAVELENGTH_UM = 0.60
MONITOR_FREQ = td.C_0 / WAVELENGTH_UM

# Broadband source
FREQ0 = 499654096666666.56
FWIDTH = 187370286250000

# Local mesh
DL = 0.0002

print("Target wavelength =", WAVELENGTH_UM, "um")
print("Monitor frequency =", MONITOR_FREQ, "Hz")

# ============================================================
# CELL 3
# READ PDB AND CENTER PROTEIN
# ============================================================

parser = PDBParser(
    QUIET=True
)

pdb_structure = parser.get_structure(
    "protein",
    PDB_FILE,
)


base_name = os.path.splitext(
    os.path.basename(PDB_FILE)
)[0]


atom_coords_A = []


for atom in pdb_structure.get_atoms():

    position = atom.get_coord()

    atom_coords_A.append(
        position
    )


atom_coords_A = np.asarray(
    atom_coords_A,
    dtype=float,
)


print(
    "Number of atoms:",
    len(atom_coords_A)
)


print()

print(
    "Original PDB bounds [Angstrom]:"
)

print(
    "min =",
    atom_coords_A.min(axis=0)
)

print(
    "max =",
    atom_coords_A.max(axis=0)
)


# ------------------------------------------------------------
# Center protein at origin
# ------------------------------------------------------------

pdb_min = atom_coords_A.min(
    axis=0
)

pdb_max = atom_coords_A.max(
    axis=0
)


protein_center_A = (
    pdb_min + pdb_max
) / 2.0


atom_coords_centered_A = (
    atom_coords_A
    - protein_center_A
)


# ------------------------------------------------------------
# Keep the original protein orientation
# Only translate the protein so that its bounding-box center
# is located at the origin. No rotation is applied.
# ------------------------------------------------------------

print()

print(
    "Removed protein center [Angstrom]:",
    protein_center_A
)


print()

print(
    "Centered bounds [Angstrom]:"
)

print(
    "min =",
    atom_coords_centered_A.min(axis=0)
)

print(
    "max =",
    atom_coords_centered_A.max(axis=0)
)

# ============================================================
# CELL 4
# PDB -> STL
# ============================================================

sphere_mesh = trimesh.creation.icosphere(
    subdivisions=2,
    radius=SPHERE_RADIUS_A,
)

all_meshes = []

for coord_A in atom_coords_centered_A:
    mesh = sphere_mesh.copy()
    mesh.apply_translation(coord_A)
    all_meshes.append(mesh)

combined = trimesh.util.concatenate(all_meshes)
combined.remove_unreferenced_vertices()

STL_FILE = os.path.join(
    DATA_DIR,
    f"{base_name}_protein_atoms.stl"
)

combined.export(STL_FILE)

print("STL generated:", STL_FILE)
print("Vertices:", len(combined.vertices))
print("Faces:", len(combined.faces))
print("Bounds [Angstrom]:")
print(combined.bounds)


# ------------------------------------------------------------
# 3D visualization
# Show protein as molecular surface
# ------------------------------------------------------------

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")

vertices = combined.vertices
faces = combined.faces

# Plot the actual protein mesh
ax.plot_trisurf(
    vertices[:, 0],
    vertices[:, 1],
    vertices[:, 2],
    triangles=faces,
    linewidth=0.0,
    antialiased=True,
    alpha=1.0,
)

# ------------------------------------------------------------
# Equal scale for x, y, z
# ------------------------------------------------------------

coord_min = vertices.min(axis=0)
coord_max = vertices.max(axis=0)

center = (coord_min + coord_max) / 2.0
max_range = np.max(coord_max - coord_min)

half_range = 0.55 * max_range

ax.set_xlim(
    center[0] - half_range,
    center[0] + half_range
)

ax.set_ylim(
    center[1] - half_range,
    center[1] + half_range
)

ax.set_zlim(
    center[2] - half_range,
    center[2] + half_range
)

ax.set_box_aspect((1, 1, 1))

# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

ax.set_xlabel("x [Å]")
ax.set_ylabel("y [Å]")
ax.set_zlabel("z [Å]")

ax.set_title("Protein molecular surface")

# Better viewing angle
ax.view_init(
    elev=20,
    azim=45
)

plt.tight_layout()
plt.show()

# ============================================================
# CELL 5
# STL -> TIDY3D
# ============================================================

protein_geometry = td.TriangleMesh.from_stl(
    filename=STL_FILE,
    scale=ANGSTROM_TO_UM,
    origin=(0, 0, 0),
)

print("Protein bounds [um]:")
print(protein_geometry.bounds)


water = td.Medium(
    name="water",
    permittivity=EPS_WATER,
)

protein_medium = td.Medium(
    name="protein",
    permittivity=EPS_PROTEIN,
)

protein_structure = td.Structure(
    name="protein_structure",
    geometry=protein_geometry,
    medium=protein_medium,
)

# ============================================================
# CELL 6
# MONITORS AND MESH
# ============================================================

Fields = td.FieldMonitor(
    name="Fields",
    center=[0, 0, 0],
    size=[5, 0, 5],
    freqs=[MONITOR_FREQ],
    colocate=True,
)

# ============================================================
# Monitor / mesh box size
# ============================================================

BOX_SIZE = 0.008  # μm = 8 nm


# ============================================================
# Field monitor
# ============================================================

fieldmonitor_1 = td.FieldMonitor(
    name="fieldmonitor_1",
    center=[0, 0, 0],
    size=[BOX_SIZE, BOX_SIZE, BOX_SIZE],
    freqs=[MONITOR_FREQ],
    colocate=True,
    apodization=td.ApodizationSpec(),
)


# ============================================================
# Permittivity monitor
# ============================================================

permittivitymonitor_2 = td.PermittivityMonitor(
    name="permittivitymonitor_2",
    center=[0, 0, 0],
    size=[BOX_SIZE, BOX_SIZE, BOX_SIZE],
    freqs=[MONITOR_FREQ],
    apodization=td.ApodizationSpec(),
)


# ============================================================
# Mesh override
# ============================================================

override_structure_0 = td.MeshOverrideStructure(
    name="override_structure_0",
    geometry=td.Box(
        center=[0, 0, 0],
        size=[BOX_SIZE, BOX_SIZE, BOX_SIZE],
    ),
    dl=[DL, DL, DL],
)

# ============================================================
# CELL 7
# BUILD SIMULATION
# ============================================================

def build_simulation(
    plane_wave1,
    plane_wave2,
):

    return td.Simulation(
        size=[0.5, 0.5, 0.5],

        boundary_spec=td.BoundarySpec(
            x=td.Boundary(
                plus=td.PML(),
                minus=td.PML(),
            ),
            y=td.Boundary(
                plus=td.PML(),
                minus=td.PML(),
            ),
            z=td.Boundary(
                plus=td.PML(
                    parameters=td.PMLParams(
                        kappa_min=1,
                        kappa_max=3,
                        alpha_order=1,
                        alpha_max=0,
                    )
                ),
                minus=td.PML(
                    parameters=td.PMLParams(
                        kappa_min=1,
                        kappa_max=3,
                        alpha_order=1,
                        alpha_max=0,
                    )
                ),
            ),
        ),

        grid_spec=td.GridSpec.auto(
            wavelength=0.4,
            override_structures=[
                override_structure_0
            ],
            min_steps_per_wvl=50,
        ),

        subpixel=td.SubpixelSpec(
            dielectric=td.PolarizedAveraging(),
            metal=td.Staircasing(),
            pec=td.PECConformal(),
        ),

        run_time=1e-13,

        medium=water,

        sources=[
            plane_wave1,
            plane_wave2,
        ],

        monitors=[
            Fields,
            fieldmonitor_1,
            permittivitymonitor_2,
        ],

        structures=[
            protein_structure
        ],
    )

# ============================================================
# CELL 8
# Ex EXCITATION
#
# Two counter-propagating plane waves along z.
# Same phase -> electric field adds at the origin,
# while the magnetic field cancels.
#
# For propagation along z:
#   pol_angle = 0  -> E along x
# ============================================================

plane_wave1_Ex = td.PlaneWave(
    name="plane_wave1_Ex",
    center=[0, 0, -0.2],
    size=[5, 5, 0],

    source_time=td.GaussianPulse(
        freq0=FREQ0,
        fwidth=FWIDTH,
    ),

    direction="+",
    pol_angle=0.0,
    angular_spec=td.FixedInPlaneKSpec(),
)


plane_wave2_Ex = td.PlaneWave(
    name="plane_wave2_Ex",
    center=[0, 0, 0.2],
    size=[5, 5, 0],

    source_time=td.GaussianPulse(
        freq0=FREQ0,
        fwidth=FWIDTH,
    ),

    direction="-",
    pol_angle=0.0,
    angular_spec=td.FixedInPlaneKSpec(),
)


sim_Ex = build_simulation(
    plane_wave1_Ex,
    plane_wave2_Ex,
)

# ============================================================
# CELL 9
# Ey EXCITATION
#
# Two counter-propagating plane waves along z.
# Same phase -> electric field adds at the origin.
#
# For propagation along z:
#   pol_angle = pi/2 -> E along y
# ============================================================

plane_wave1_Ey = td.PlaneWave(
    name="plane_wave1_Ey",
    center=[0, 0, -0.2],
    size=[5, 5, 0],

    source_time=td.GaussianPulse(
        freq0=FREQ0,
        fwidth=FWIDTH,
    ),

    direction="+",
    pol_angle=np.pi / 2,
    angular_spec=td.FixedInPlaneKSpec(),
)


plane_wave2_Ey = td.PlaneWave(
    name="plane_wave2_Ey",
    center=[0, 0, 0.2],
    size=[5, 5, 0],

    source_time=td.GaussianPulse(
        freq0=FREQ0,
        fwidth=FWIDTH,
    ),

    direction="-",
    pol_angle=np.pi / 2,
    angular_spec=td.FixedInPlaneKSpec(),
)


sim_Ey = build_simulation(
    plane_wave1_Ey,
    plane_wave2_Ey,
)

# ============================================================
# CELL 10
# Ez EXCITATION
#
# A plane wave cannot have E parallel to propagation.
# Therefore use counter-propagation along x.
#
# For propagation along x:
#   pol_angle = pi/2 -> E along z
#
# Same phase -> electric field adds at the origin.
# ============================================================

plane_wave1_Ez = td.PlaneWave(
    name="plane_wave1_Ez",
    center=[-0.2, 0, 0],
    size=[0, 5, 5],

    source_time=td.GaussianPulse(
        freq0=FREQ0,
        fwidth=FWIDTH,
    ),

    direction="+",
    pol_angle=np.pi / 2,
    angular_spec=td.FixedInPlaneKSpec(),
)


plane_wave2_Ez = td.PlaneWave(
    name="plane_wave2_Ez",
    center=[0.2, 0, 0],
    size=[0, 5, 5],

    source_time=td.GaussianPulse(
        freq0=FREQ0,
        fwidth=FWIDTH,
    ),

    direction="-",
    pol_angle=np.pi / 2,
    angular_spec=td.FixedInPlaneKSpec(),
)


sim_Ez = build_simulation(
    plane_wave1_Ez,
    plane_wave2_Ez,
)

# ============================================================
# CELL 11
# GEOMETRY VISUALIZATION
# ============================================================

fig, ax = plt.subplots(figsize=(6, 5))

sim_Ex.plot_eps(
    z=0,
    ax=ax,
)

ax.set_title("Permittivity: xy plane, z = 0")

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(6, 5))

sim_Ex.plot_eps(
    y=0,
    ax=ax,
)

ax.set_title("Permittivity: xz plane, y = 0")

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(6, 5))

sim_Ex.plot_eps(
    x=0,
    ax=ax,
)

ax.set_title("Permittivity: yz plane, x = 0")

plt.tight_layout()
plt.show()

# ============================================================
# CELL 12
# RUN SIMULATIONS
# ============================================================

sim_data_Ex = web.run(
    sim_Ex,
    task_name="GFP_670nm_Ex_FDTD",
    verbose=True,
)


sim_data_Ey = web.run(
    sim_Ey,
    task_name="GFP_670nm_Ey_FDTD",
    verbose=True,
)


sim_data_Ez = web.run(
    sim_Ez,
    task_name="GFP_670nm_Ez_FDTD",
    verbose=True,
)

# ============================================================
# CELL 13
# ELECTRIC FIELD VISUALIZATION
# ============================================================

sim_data_Ex.plot_field(
    "Fields",
    field_name="Ex",
    val="abs",
    f=MONITOR_FREQ,
)

plt.title(
    "|Ex| at 670 nm"
)

plt.show()


sim_data_Ey.plot_field(
    "Fields",
    field_name="Ey",
    val="abs",
    f=MONITOR_FREQ,
)

plt.title(
    "|Ey| at 670 nm"
)

plt.show()


sim_data_Ez.plot_field(
    "Fields",
    field_name="Ez",
    val="abs",
    f=MONITOR_FREQ,
)

plt.title(
    "|Ez| at 670 nm"
)

plt.show()

# ============================================================
# CELL 14
# DIPOLE MOMENT FROM div(E)
# ============================================================

def calculate_dipole_from_divE(
    sim_data,
):

    field_data = sim_data[
        "fieldmonitor_1"
    ]

    Ex = field_data.Ex.isel(f=0)
    Ey = field_data.Ey.isel(f=0)
    Ez = field_data.Ez.isel(f=0)

    divE = (
        Ex.differentiate("x")
        + Ey.differentiate("y")
        + Ez.differentiate("z")
    )

    x = divE.x
    y = divE.y
    z = divE.z

    px_density = x * divE
    py_density = y * divE
    pz_density = z * divE

    px = eps0 * px_density.integrate(
        coord=["x", "y", "z"]
    )

    py = eps0 * py_density.integrate(
        coord=["x", "y", "z"]
    )

    pz = eps0 * pz_density.integrate(
        coord=["x", "y", "z"]
    )

    return (
        px.values,
        py.values,
        pz.values,
    )

# ============================================================
# CELL 15
# DIPOLE RESULTS
# ============================================================

px_Ex, py_Ex, pz_Ex = (
    calculate_dipole_from_divE(
        sim_data_Ex
    )
)


px_Ey, py_Ey, pz_Ey = (
    calculate_dipole_from_divE(
        sim_data_Ey
    )
)


px_Ez, py_Ez, pz_Ez = (
    calculate_dipole_from_divE(
        sim_data_Ez
    )
)


print("Ex excitation")
print("px =", px_Ex)
print("py =", py_Ex)
print("pz =", pz_Ex)

print()

print("Ey excitation")
print("px =", px_Ey)
print("py =", py_Ey)
print("pz =", pz_Ey)

print()

print("Ez excitation")
print("px =", px_Ez)
print("py =", py_Ez)
print("pz =", pz_Ez)

# ============================================================
# CELL 16
# alpha_ee
#
# Diagonal electric polarizability:
#
# alpha_ee_xx = p_x / E_x
# alpha_ee_yy = p_y / E_y
# alpha_ee_zz = p_z / E_z
#
# Coordinate integration is performed in um, so retain the
# original 1e-18 conversion factor.
# ============================================================

E_ext = E_EXT


alpha_ee_x = (
    np.abs(px_Ex)
    * 1e-18
    / E_ext
)

alpha_ee_y = (
    np.abs(py_Ey)
    * 1e-18
    / E_ext
)

alpha_ee_z = (
    np.abs(pz_Ez)
    * 1e-18
    / E_ext
)


print("alpha_ee_x =", alpha_ee_x)
print("alpha_ee_y =", alpha_ee_y)
print("alpha_ee_z =", alpha_ee_z)


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

labels = [
    "alpha_ee_x",
    "alpha_ee_y",
    "alpha_ee_z",
]

values = [
    alpha_ee_x,
    alpha_ee_y,
    alpha_ee_z,
]


fig, ax = plt.subplots(
    figsize=(6, 4)
)

ax.bar(
    labels,
    values,
)

ax.set_ylabel(
    "alpha_ee [F m^2]"
)

ax.set_title(
    "Electric polarizability at 670 nm"
)

ax.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()
plt.show()
