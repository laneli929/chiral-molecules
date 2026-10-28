"""EM 磁电响应：已有一对位移 PDB -> 小球 STL -> 6 个 Tidy3D 任务 -> 两类差分。
依赖：python -m pip install numpy biopython trimesh "tidy3d[trimesh]"
可直接在 Jupyter 中运行整个脚本，或终端执行：python tidy3d_em_existing_pdb_pair.py
先修改开头的 PDB 路径和实际位移幅度；RUN_CLOUD=False 时只准备仿真输入。

只需 plus/minus 两个 PDB，不需要 ProDy、metadata.json 或 reference.pdb。
假定两者是同一坐标系中沿同一单位模态的 ±beta 位移，原子身份与顺序一致。
保留原代码的小球网格拼接、双平面波、div(E) 积分和固定 H_EXT 归一化。
结果沿用该极化率估计，未进行空背景场标定；拼接小球不是布尔并集。
定义 p = alpha_em H，此处不是 m/E 或 m/H。
H_EXT 默认 0.27 A/μm，复现参考代码的标量幅度归一化；它不是已标定的复数外场。
严格的复数张量（尤其方向平均）需各激励的有符号复数 H_inc，不能把 |H| 当 H。
即使中心处电场抵消，有限尺寸结构中的电场梯度也可能贡献响应；本方法不是完整
多极/双各向异性参数反演。单位使用 s·m²（p/H），不同于 EE 的 s·m²。
官方单位：https://docs.flexcompute.com/projects/tidy3d/en/stable/api/_autosummary/tidy3d.PointCloudFieldData.html
"""
import csv
import warnings
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import trimesh
import tidy3d as td
import tidy3d.web as web
from Bio.PDB import PDBParser

# ==================== 文件与 beta 参数 ====================
PDB_DIR = Path("./pdb_pair")
PLUS_PDB = PDB_DIR / "plus.pdb"
MINUS_PDB = PDB_DIR / "minus.pdb"

# 必须与生成这两个 PDB 时的真实步长一致，不能仅修改这里改变已有结构。
STEP_DEFINITION = "beta"         # "beta" 或 "rms"，二选一
BETA_A = 1.0                     # 示例值；STEP_DEFINITION="beta" 时使用
RMS_STEP_A = 0.025                # STEP_DEFINITION="rms" 时使用；beta=RMS*sqrt(N)
EXPECTED_ATOM_COUNT = None       # 可设为 1260；None 表示不固定原子数
PAIR_DISTANCE_WARNING_FRACTION = 0.05

OUTPUT_ROOT = Path("./data_em_difference")
TASK_PREFIX = "1MBN_EM"
PDB_MODEL_INDEX = 0              # Bio.PDB 模型编号从 0 开始

# ==================== 几何、材料、光源、网格 ====================
SPHERE_RADIUS_A = 2.0
SPHERE_SUBDIVISIONS = 2
EPS_WATER = 1.33**2
EPS_PROTEIN = 1.6**2

# 分别对应 Hx、Hy、Hz 激励，单位 A/μm。默认与参考代码的正数 0.27 相同。
# 若已知背景场的有符号复数值（包含相位），在这里逐项填写。
# 固定正数归一化下的复数差分/迹平均为参考方法估计，不是经标定的张量。
H_EXT_A_PER_UM = (0.27 + 0j, 0.27 + 0j, 0.27 + 0j)
WAVELENGTH_UM = 0.60
SOURCE_FREQ_HZ = 499654096666666.56 # 保留原代码；也可设 None 自动按波长计算
SOURCE_FWIDTH_HZ = 187370286250000.0
SOURCE_AMPLITUDE = 1.0
SOURCE_PHASE_RAD = 0.0
COUNTER_SOURCE_PHASE_DIFFERENCE_RAD = np.pi
SOURCE_TRANSVERSE_SIZE_UM = 5.0  # 与参考 EM 代码一致
SOURCE_OFFSET_UM = 0.2           # 两个源位于 ±此距离

SIM_SIZE_UM = (0.5, 0.5, 0.5)   # 整个仿真域三条边长，单位 μm
BOX_SIZE_UM = (0.008, 0.008, 0.008) # 监视器/积分盒三条边长
DL_UM = 0.0002                  # 积分盒内网格步长
BOX_MARGIN_CELLS = 2
GRID_WAVELENGTH_UM = 0.4
MIN_STEPS_PER_WVL = 50
RUN_TIME_S = 1e-13
SHUTOFF = 1e-5
PML_LAYERS = 12
Z_PML_KAPPA_MIN = 1.0
Z_PML_KAPPA_MAX = 3.0
Z_PML_ALPHA_ORDER = 1
Z_PML_ALPHA_MAX = 0.0
RUN_CLOUD = True                # False 时只准备输入；不提交云端任务

# 单位常量
ANGSTROM_TO_UM = 1e-4
EPS0 = 8.8541878128e-12
AXES = "xyz"
MONITOR_NAME = "fieldmonitor_1"



def read_coords(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    structure = PDBParser(QUIET=True).get_structure(path.stem, str(path))
    atoms = list(structure[PDB_MODEL_INDEX].get_atoms())
    if not atoms:
        raise ValueError(f"没有原子：{path}")
    ids = [(a.get_parent().get_parent().id, a.get_parent().id,
            a.get_name(), a.get_altloc()) for a in atoms]
    return np.asarray([a.coord for a in atoms], dtype=float), ids


def load_pair():
    plus, plus_ids = read_coords(PLUS_PDB)
    minus, minus_ids = read_coords(MINUS_PDB)
    if plus_ids != minus_ids:
        raise ValueError("两个 PDB 的原子身份或顺序不同")
    n_atoms = len(plus)
    if EXPECTED_ATOM_COUNT is not None and n_atoms != EXPECTED_ATOM_COUNT:
        raise ValueError(f"实际原子数 {n_atoms} != {EXPECTED_ATOM_COUNT}")
    if STEP_DEFINITION == "beta":
        beta = float(BETA_A)
    elif STEP_DEFINITION == "rms":
        beta = float(RMS_STEP_A)*np.sqrt(n_atoms)
    else:
        raise ValueError('STEP_DEFINITION 必须为 "beta" 或 "rms"')
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("位移幅度必须为有限正数")
    if not np.all(np.isfinite(plus)) or not np.all(np.isfinite(minus)):
        raise ValueError("PDB 包含非有限坐标")
    measured_distance = float(np.linalg.norm(plus-minus))
    if measured_distance == 0:
        raise ValueError("两个 PDB 坐标相同，无法计算扰动差分")
    relative_error = abs(measured_distance-2*beta)/(2*beta)
    if relative_error > PAIR_DISTANCE_WARNING_FRACTION:
        warnings.warn(f"实际 3N 维构象距离 {measured_distance:.8g} Å 与 2β={2*beta:.8g} Å "
                      f"相差 {relative_error:.1%}。请核对步长、原子集合和坐标系；"
                      "本程序仍以设置的 2β 为差分分母。")
    # 对称位移满足 (r_plus+r_minus)/2 ≈ r0；只计算一个共同平移中心。
    # 不分别居中、旋转或对齐两个构象，以保留原始模态位移。
    midpoint = (plus+minus)/2
    center = (midpoint.min(axis=0)+midpoint.max(axis=0))/2
    metadata = {
        "beta_A": beta, "step_definition": STEP_DEFINITION,
        "rms_step_A": beta/np.sqrt(n_atoms), "n_atoms": n_atoms,
        "difference_denominator_A": 2*beta,
        "measured_pair_distance_A": measured_distance,
        "relative_pair_distance_error": relative_error,
        "center_A": center.tolist(),
        "plus_pdb": str(PLUS_PDB.resolve()), "minus_pdb": str(MINUS_PDB.resolve()),
    }
    print(json.dumps(metadata, indent=2))
    return {"plus": plus-center, "minus": minus-center}, beta, metadata


def make_structure(coords_A, label, out):
    extent = (np.max(np.abs(coords_A), axis=0) + SPHERE_RADIUS_A)*ANGSTROM_TO_UM
    if np.any(extent >= np.asarray(BOX_SIZE_UM)/2 - BOX_MARGIN_CELLS*DL_UM):
        raise ValueError(f"{label} 超出积分盒或离边缘太近，请增大 BOX_SIZE_UM")
    sphere = trimesh.creation.icosphere(subdivisions=SPHERE_SUBDIVISIONS,
                                        radius=SPHERE_RADIUS_A)
    meshes = []
    for coord in coords_A:
        mesh = sphere.copy()
        mesh.apply_translation(coord)
        meshes.append(mesh)
    combined = trimesh.util.concatenate(meshes)
    combined.remove_unreferenced_vertices()
    path = out / f"{label}.stl"
    combined.export(str(path))
    geometry = td.TriangleMesh.from_stl(filename=str(path), scale=ANGSTROM_TO_UM,
                                        origin=(0, 0, 0))
    return td.Structure(name="protein_structure", geometry=geometry,
                        medium=td.Medium(permittivity=EPS_PROTEIN))

def make_sources(axis):
    # 完全沿用参考 EM 代码：Hx:(z传播,π/2)，Hy:(z传播,0)，Hz:(x传播,0)。
    # 两束相向波的电场源相位相差 π，与原 EE 版本同相源不同。
    if axis not in AXES:
        raise ValueError("axis 必须为 x/y/z")
    propagation_axis = 0 if axis == "z" else 2
    pol_angle = np.pi/2 if axis == "x" else 0.0
    size = [SOURCE_TRANSVERSE_SIZE_UM]*3
    size[propagation_axis] = 0
    freq0 = td.C_0/WAVELENGTH_UM if SOURCE_FREQ_HZ is None else SOURCE_FREQ_HZ
    sources = []
    for index, (position, direction) in enumerate(
            ((-SOURCE_OFFSET_UM, "+"), (SOURCE_OFFSET_UM, "-"))):
        center = [0.0]*3
        center[propagation_axis] = position
        phase = SOURCE_PHASE_RAD + index*COUNTER_SOURCE_PHASE_DIFFERENCE_RAD
        sources.append(td.PlaneWave(
            name=f"plane_wave{index+1}_H{axis}",
            center=center, size=size, direction=direction, pol_angle=pol_angle,
            source_time=td.GaussianPulse(freq0=freq0, fwidth=SOURCE_FWIDTH_HZ,
                                         amplitude=SOURCE_AMPLITUDE, phase=phase),
            angular_spec=td.FixedInPlaneKSpec()))
    return sources

def build_simulation(structure, axis, grid):
    z_pml = td.PML(num_layers=PML_LAYERS, parameters=td.PMLParams(
        kappa_min=Z_PML_KAPPA_MIN, kappa_max=Z_PML_KAPPA_MAX,
        alpha_order=Z_PML_ALPHA_ORDER, alpha_max=Z_PML_ALPHA_MAX))
    return td.Simulation(
        size=SIM_SIZE_UM, medium=td.Medium(permittivity=EPS_WATER),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=PML_LAYERS),
            y=td.Boundary.pml(num_layers=PML_LAYERS),
            z=td.Boundary(plus=z_pml, minus=z_pml)),
        grid_spec=grid,
        subpixel=td.SubpixelSpec(dielectric=td.PolarizedAveraging(),
                                metal=td.Staircasing(), pec=td.PECConformal()),
        run_time=RUN_TIME_S, shutoff=SHUTOFF,
        sources=make_sources(axis), structures=[structure],
        monitors=[td.FieldMonitor(name=MONITOR_NAME, center=(0, 0, 0),
                                  size=BOX_SIZE_UM, freqs=[td.C_0/WAVELENGTH_UM],
                                  colocate=True)])

def calculate_original_dipole_integral(sim_data):
    fields = sim_data[MONITOR_NAME]
    ex, ey, ez = (getattr(fields, "E"+a).isel(f=0) for a in AXES)
    div_e = ex.differentiate("x") + ey.differentiate("y") + ez.differentiate("z")
    return EPS0*np.array([
        complex((div_e[a]*div_e).integrate(coord=list(AXES)).values.item())
        for a in AXES
    ])

def save_results(alpha, out, beta):
    delta_complex = alpha["plus"] - alpha["minus"]
    # 绝对值的差：可以为负；与 abs(delta_complex) 是不同的量。
    delta_magnitude = np.abs(alpha["plus"]) - np.abs(alpha["minus"])
    derivative_complex = delta_complex/(2*beta)
    derivative_magnitude = delta_magnitude/(2*beta)
    with (out / "em_polarizabilities.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["structure", "component", "alpha_real_s_m2",
                    "alpha_imag_s_m2", "alpha_abs_s_m2"])
        for label, values in alpha.items():
            for a, value in zip(AXES, values):
                w.writerow([label, a+a, value.real, value.imag, abs(value)])
    with (out / "em_complex_difference.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beta_A", "distance_2beta_A", "component",
                    "delta_complex_real_s_m2", "delta_complex_imag_s_m2",
                    "abs_of_complex_difference_s_m2",
                    "derivative_complex_real_s_m2_per_A",
                    "derivative_complex_imag_s_m2_per_A"])
        for i, a in enumerate(AXES):
            w.writerow([beta, 2*beta, a+a, delta_complex[i].real,
                        delta_complex[i].imag, abs(delta_complex[i]),
                        derivative_complex[i].real, derivative_complex[i].imag])
    with (out / "em_absolute_value_difference.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beta_A", "distance_2beta_A", "component",
                    "abs_alpha_plus_s_m2", "abs_alpha_minus_s_m2",
                    "difference_of_abs_s_m2", "derivative_of_abs_s_m2_per_A"])
        for i, a in enumerate(AXES):
            w.writerow([beta, 2*beta, a+a, abs(alpha["plus"][i]),
                        abs(alpha["minus"][i]), delta_magnitude[i], derivative_magnitude[i]])
            print(f"{a+a}: Δalpha_em = {delta_complex[i]:.8e}; "
                  f"Δalpha_em/(2β) = {derivative_complex[i]:.8e}")
            print(f"    Δ|alpha_em| = {delta_magnitude[i]:.8e}; "
                  f"Δ|alpha_em|/(2β) = {derivative_magnitude[i]:.8e}")
    # 两种方向平均都输出，避免将“平均模长”与“平均复数的模长”混淆。
    mean_plus = np.mean(alpha["plus"])
    mean_minus = np.mean(alpha["minus"])
    mean_delta = np.mean(delta_complex)
    mean_abs_delta = np.mean(delta_magnitude)
    with (out / "em_direction_averages.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "real", "imag", "unit"])
        for name, value, unit in [
            ("mean_complex_plus", mean_plus, "s*m^2"),
            ("mean_complex_minus", mean_minus, "s*m^2"),
            ("mean_complex_difference", mean_delta, "s*m^2"),
            ("mean_complex_derivative", mean_delta/(2*beta), "s*m^2/Angstrom"),
            ("mean_magnitude_plus", np.mean(np.abs(alpha["plus"])), "s*m^2"),
            ("mean_magnitude_minus", np.mean(np.abs(alpha["minus"])), "s*m^2"),
            ("mean_of_magnitude_differences", mean_abs_delta, "s*m^2"),
            ("mean_of_magnitude_derivatives", mean_abs_delta/(2*beta), "s*m^2/Angstrom"),
            ("abs_of_mean_complex_difference", abs(mean_delta), "s*m^2"),
        ]:
            value = complex(value)
            w.writerow([name, value.real, value.imag, unit])
    print(f"三方向平均 Δalpha_em = {mean_delta:.8e}")
    print(f"三方向平均 Δalpha_em/(2β) = {mean_delta/(2*beta):.8e}")
    print(f"三方向平均 Δ|alpha_em| = {mean_abs_delta:.8e}")
    print(f"三方向平均 Δ|alpha_em|/(2β) = {mean_abs_delta/(2*beta):.8e}")
    np.savez(out / "em_differences.npz",
             mean_delta_complex=mean_delta, mean_delta_magnitude=mean_abs_delta,
             mean_derivative_complex=mean_delta/(2*beta),
             mean_derivative_magnitude=mean_abs_delta/(2*beta), beta_A=beta,
             alpha_plus=alpha["plus"], alpha_minus=alpha["minus"],
             delta_complex=delta_complex, delta_magnitude=delta_magnitude,
             derivative_complex=derivative_complex, derivative_magnitude=derivative_magnitude)


def main():
    if (np.any(np.asarray(BOX_SIZE_UM) <= 0) or
            np.any(np.asarray(BOX_SIZE_UM) >= np.asarray(SIM_SIZE_UM))):
        raise ValueError("BOX_SIZE_UM 必须为正且小于仿真域")
    if not max(BOX_SIZE_UM)/2 < SOURCE_OFFSET_UM < min(SIM_SIZE_UM)/2:
        raise ValueError("源必须位于积分盒外、仿真域内")
    h_ext = np.asarray(H_EXT_A_PER_UM, dtype=complex)
    if h_ext.shape != (3,) or not np.all(np.isfinite(h_ext)) or np.any(np.abs(h_ext) == 0):
        raise ValueError("H_EXT_A_PER_UM 必须包含三个有限且非零的复数")
    coords, beta, metadata = load_pair()
    metadata.update(response="alpha_em = p/H", alpha_unit="s*m^2",
                    normalization="user-supplied fixed complex H; uncalibrated by default")
    # 每次运行使用独立目录，避免不同参数的结果互相覆盖。
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = OUTPUT_ROOT / f"beta{beta:g}_{run_id}"
    out.mkdir(parents=True, exist_ok=False)
    config = {k: v for k, v in globals().items() if k.isupper()}
    (out / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
    (out / "input_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    structures = {label: make_structure(r, label, out) for label, r in coords.items()}

    override = td.MeshOverrideStructure(
        geometry=td.Box(center=(0, 0, 0), size=BOX_SIZE_UM), dl=(DL_UM,)*3)
    auto_grid = td.GridSpec.auto(wavelength=GRID_WAVELENGTH_UM,
                                override_structures=[override],
                                min_steps_per_wvl=MIN_STEPS_PER_WVL)
    # 从 plus 构象产生一次网格；全部 6 个任务使用完全相同的网格。
    template = build_simulation(structures["plus"], "x", auto_grid)
    fixed_grid = td.GridSpec.from_grid(template.grid)
    simulations = {}
    for label, structure in structures.items():
        for axis in AXES:
            sim = build_simulation(structure, axis, fixed_grid)
            sim.to_file(str(out / f"simulation_{label}_H{axis}.json"))
            simulations[(label, axis)] = sim
    print(f"积分盒体积 = {np.prod(BOX_SIZE_UM):.8g} μm³；输出目录：{out.resolve()}")
    if not RUN_CLOUD:
        print("输入文件已准备好；未提交云端仿真。")
        return

    alpha = {label: np.zeros(3, dtype=complex) for label in structures}
    print("开始运行 2 个构象 × 3 个方向 = 6 个 Tidy3D 云端任务。")
    for label in structures:
        for i, axis in enumerate(AXES):
            data = web.run(simulations[(label, axis)],
                           task_name=f"{TASK_PREFIX}_{run_id}_{label}_H{axis}",
                           path=str(out / f"{label}_H{axis}.hdf5"), verbose=True)
            raw = calculate_original_dipole_integral(data)
            # r 和积分坐标单位 μm；E 单位 V/μm。
            # raw 转为偶极矩 C·m 乘 1e-12，H_EXT 从 A/μm 转 A/m 乘 1e6。
            # 因而 alpha_em = raw * 1e-18 / H_EXT，单位 s·m²。
            alpha[label][i] = raw[i]*1e-18/H_EXT_A_PER_UM[i]
            print(f"{label}, alpha_em_{axis}{axis} = {alpha[label][i]:.8e} s·m²")
            np.savez(out / f"partial_{label}_H{axis}.npz",
                     alpha_diagonal=alpha[label][i], raw_dipole_integral=raw)
    save_results(alpha, out, beta)
    print(f"完成，结果保存在：{out.resolve()}")

if __name__ == "__main__":
    main()
