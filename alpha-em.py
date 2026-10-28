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

PDB_FILE = "1lst.pdb"

DATA_DIR = "./data"
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

# Geometry
SPHERE_RADIUS_A = 2.0
ANGSTROM_TO_UM = 1e-4

# Materials
EPS_WATER = 1.33**2
EPS_PROTEIN = 1.6**2

# External magnetic field
H_EXT = 0.27

# Vacuum permittivity
eps0 = 8.8541878128e-12

# Target wavelength: retain the original fast value (1.000 um)
WAVELENGTH_UM = 0.6
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

parser = PDBParser(QUIET=True)

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
    atom_coords_A.append(position)

atom_coords_A = np.asarray(
    atom_coords_A,
    dtype=float,
)

print("Number of atoms:", len(atom_coords_A))

print()
print("Original PDB bounds [Angstrom]:")
print("min =", atom_coords_A.min(axis=0))
print("max =", atom_coords_A.max(axis=0))

# ------------------------------------------------------------
# Center protein at origin
# ------------------------------------------------------------

pdb_min = atom_coords_A.min(axis=0)
pdb_max = atom_coords_A.max(axis=0)

protein_center_A = (pdb_min + pdb_max) / 2.0

atom_coords_centered_A = (
    atom_coords_A - protein_center_A
)

# Keep the original protein orientation.
# Only translate the bounding-box center to the origin.
# No rotation is applied.

print()
print(
    "Removed protein center [Angstrom]:",
    protein_center_A,
)

print()
print("Centered bounds [Angstrom]:")
print("min =", atom_coords_centered_A.min(axis=0))
print("max =", atom_coords_centered_A.max(axis=0))


# ============================================================
# CELL 4
# PDB -> STL
# Same modeling method as the 2lao code:
# create an icosphere at every atom and concatenate the meshes
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
    f"{base_name}_protein_atoms.stl",
)

combined.export(STL_FILE)

print("STL generated:", STL_FILE)
print("Vertices:", len(combined.vertices))
print("Faces:", len(combined.faces))
print("Bounds [Angstrom]:")
print(combined.bounds)

# ------------------------------------------------------------
# 3D visualization
# ------------------------------------------------------------

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")

vertices = combined.vertices
faces = combined.faces

ax.plot_trisurf(
    vertices[:, 0],
    vertices[:, 1],
    vertices[:, 2],
    triangles=faces,
    linewidth=0.0,
    antialiased=True,
    alpha=1.0,
)

# Equal scale for x, y, z
coord_min = vertices.min(axis=0)
coord_max = vertices.max(axis=0)

center = (coord_min + coord_max) / 2.0
max_range = np.max(coord_max - coord_min)
half_range = 0.55 * max_range

ax.set_xlim(
    center[0] - half_range,
    center[0] + half_range,
)
ax.set_ylim(
    center[1] - half_range,
    center[1] + half_range,
)
ax.set_zlim(
    center[2] - half_range,
    center[2] + half_range,
)

ax.set_box_aspect((1, 1, 1))

ax.set_xlabel("x [Å]")
ax.set_ylabel("y [Å]")
ax.set_zlabel("z [Å]")
ax.set_title("Protein molecular surface")

ax.view_init(elev=20, azim=45)

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

# Monitor / mesh box size: original fast value
BOX_SIZE = 0.008  # 

fieldmonitor_1 = td.FieldMonitor(
    name="fieldmonitor_1",
    center=[0, 0, 0],
    size=[BOX_SIZE, BOX_SIZE, BOX_SIZE],
    freqs=[MONITOR_FREQ],
    colocate=True,
    apodization=td.ApodizationSpec(),
)

permittivitymonitor_2 = td.PermittivityMonitor(
    name="permittivitymonitor_2",
    center=[0, 0, 0],
    size=[BOX_SIZE, BOX_SIZE, BOX_SIZE],
    freqs=[MONITOR_FREQ],
    apodization=td.ApodizationSpec(),
)

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

def build_simulation(plane_wave1, plane_wave2):
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
                    ),
                ),
                minus=td.PML(
                    parameters=td.PMLParams(
                        kappa_min=1,
                        kappa_max=3,
                        alpha_order=1,
                        alpha_max=0,
                    ),
                ),
            ),
        ),

        grid_spec=td.GridSpec.auto(
            wavelength=0.4,
            override_structures=[
                override_structure_0,
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
            protein_structure,
        ],
    )


# ============================================================
# CELL 8
# Hx EXCITATION
# ============================================================

plane_wave1_Hx = td.PlaneWave(
    name="plane_wave1_Hx",
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

plane_wave2_Hx = td.PlaneWave(
    name="plane_wave2_Hx",
    center=[0, 0, 0.2],
    size=[5, 5, 0],

    source_time=td.GaussianPulse(
        phase=np.pi,
        freq0=FREQ0,
        fwidth=FWIDTH,
    ),

    direction="-",
    pol_angle=np.pi / 2,
    angular_spec=td.FixedInPlaneKSpec(),
)

sim_Hx = build_simulation(
    plane_wave1_Hx,
    plane_wave2_Hx,
)


# ============================================================
# CELL 9
# Hy EXCITATION
# ============================================================

plane_wave1_Hy = td.PlaneWave(
    name="plane_wave1_Hy",
    center=[0, 0, -0.2],
    size=[5, 5, 0],

    source_time=td.GaussianPulse(
        freq0=FREQ0,
        fwidth=FWIDTH,
    ),

    direction="+",
    angular_spec=td.FixedInPlaneKSpec(),
)

plane_wave2_Hy = td.PlaneWave(
    name="plane_wave2_Hy",
    center=[0, 0, 0.2],
    size=[5, 5, 0],

    source_time=td.GaussianPulse(
        phase=np.pi,
        freq0=FREQ0,
        fwidth=FWIDTH,
    ),

    direction="-",
    angular_spec=td.FixedInPlaneKSpec(),
)

sim_Hy = build_simulation(
    plane_wave1_Hy,
    plane_wave2_Hy,
)


# ============================================================
# CELL 10
# Hz EXCITATION
# ============================================================

plane_wave1_Hz = td.PlaneWave(
    name="plane_wave1_Hz",
    center=[-0.2, 0, 0],
    size=[0, 5, 5],

    source_time=td.GaussianPulse(
        freq0=FREQ0,
        fwidth=FWIDTH,
    ),

    direction="+",
    angular_spec=td.FixedInPlaneKSpec(),
)

plane_wave2_Hz = td.PlaneWave(
    name="plane_wave2_Hz",
    center=[0.2, 0, 0],
    size=[0, 5, 5],

    source_time=td.GaussianPulse(
        phase=np.pi,
        freq0=FREQ0,
        fwidth=FWIDTH,
    ),

    direction="-",
    angular_spec=td.FixedInPlaneKSpec(),
)

sim_Hz = build_simulation(
    plane_wave1_Hz,
    plane_wave2_Hz,
)


# ============================================================
# CELL 11
# GEOMETRY VISUALIZATION
# ============================================================

fig, ax = plt.subplots(figsize=(6, 5))

sim_Hx.plot_eps(
    z=0,
    ax=ax,
)

ax.set_title("Permittivity: xy plane, z = 0")

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(6, 5))

sim_Hx.plot_eps(
    y=0,
    ax=ax,
)

ax.set_title("Permittivity: xz plane, y = 0")

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(6, 5))

sim_Hx.plot_eps(
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

sim_data_Hx = web.run(
    sim_Hx,
    task_name="GFP_670nm_Hx_FDTD",
    verbose=True,
)

sim_data_Hy = web.run(
    sim_Hy,
    task_name="GFP_670nm_Hy_FDTD",
    verbose=True,
)

sim_data_Hz = web.run(
    sim_Hz,
    task_name="GFP_670nm_Hz_FDTD",
    verbose=True,
)


# ============================================================
# CELL 13
# MAGNETIC FIELD VISUALIZATION
# ============================================================

sim_data_Hx.plot_field(
    "Fields",
    field_name="Hx",
    val="abs",
    f=MONITOR_FREQ,
)

plt.title("|Hx| at 670 nm")
plt.show()


sim_data_Hy.plot_field(
    "Fields",
    field_name="Hy",
    val="abs",
    f=MONITOR_FREQ,
)

plt.title("|Hy| at 670 nm")
plt.show()


sim_data_Hz.plot_field(
    "Fields",
    field_name="Hz",
    val="abs",
    f=MONITOR_FREQ,
)

plt.title("|Hz| at 670 nm")
plt.show()


# ============================================================
# CELL 14
# DIPOLE MOMENT FROM div(E)
# ============================================================

def calculate_dipole_from_divE(sim_data):
    field_data = sim_data["fieldmonitor_1"]

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
        coord=["x", "y", "z"],
    )

    py = eps0 * py_density.integrate(
        coord=["x", "y", "z"],
    )

    pz = eps0 * pz_density.integrate(
        coord=["x", "y", "z"],
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

px_Hx, py_Hx, pz_Hx = calculate_dipole_from_divE(
    sim_data_Hx,
)

px_Hy, py_Hy, pz_Hy = calculate_dipole_from_divE(
    sim_data_Hy,
)

px_Hz, py_Hz, pz_Hz = calculate_dipole_from_divE(
    sim_data_Hz,
)

print("Hx excitation")
print("px =", px_Hx)
print("py =", py_Hx)
print("pz =", pz_Hx)

print()

print("Hy excitation")
print("px =", px_Hy)
print("py =", py_Hy)
print("pz =", pz_Hy)

print()

print("Hz excitation")
print("px =", px_Hz)
print("py =", py_Hz)
print("pz =", pz_Hz)


# ============================================================
# CELL 16
# alpha_em
# ============================================================

H_ext = H_EXT

alpha_em_x = (
    np.abs(px_Hx)
    * 1e-18
    / H_ext
)

alpha_em_y = (
    np.abs(py_Hy)
    * 1e-18
    / H_ext
)

alpha_em_z = (
    np.abs(pz_Hz)
    * 1e-18
    / H_ext
)

print("alpha_em_x =", alpha_em_x)
print("alpha_em_y =", alpha_em_y)
print("alpha_em_z =", alpha_em_z)

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

labels = [
    "alpha_em_x",
    "alpha_em_y",
    "alpha_em_z",
]

values = [
    alpha_em_x,
    alpha_em_y,
    alpha_em_z,
]

fig, ax = plt.subplots(figsize=(6, 4))

ax.bar(labels, values)

ax.set_ylabel("alpha_em")
ax.set_title("Magnetoelectric polarizability at 670 nm")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
