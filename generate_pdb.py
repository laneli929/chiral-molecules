"""第一步：ProDy 生成单一模态的 ±beta PDB。
安装：python -m pip install prody numpy scipy
运行：python 01_generate_pdb_pair.py
所有相对路径以当前工作目录为准。重新运行将更新 OUTPUT_DIR 中的生成文件。
"""
import json
import hashlib
import warnings
from pathlib import Path
import numpy as np
import prody as pd
from scipy.sparse.csgraph import connected_components

# ==================== 可调参数 ====================
INPUT_PDB = Path("./1mbn_reference.pdb")
OUTPUT_DIR = Path("./pdb_pair")
TASK_PREFIX = "1MBN"
PDB_MODEL = 1
PDB_ALTLOC = "A"
ATOM_SELECTION = "not hydrogen"  # 保留输入中的 HEM/OH 等非氢原子
EXPECTED_ATOM_COUNT = None       # 可设为 1260
MODE_NUMBER = 1                  # 第一个非零模态，编号从 1 开始
BETA_A = 1.0                     # Å；3N 维单位模态幅度，不是原子 RMS
ANM_CUTOFF_A = 7.9
ANM_GAMMA = 1.0
MODE_SIGN = 1
PDB_ERROR_WARNING_FRACTION = 0.05

def read_pdb(path):
    atoms = pd.parsePDB(str(path.resolve()), model=PDB_MODEL,
                        altloc=PDB_ALTLOC)
    if atoms is None:
        raise ValueError(f"无法读取 PDB：{path}")
    return atoms

def atom_ids(atoms):
    return list(zip(atoms.getChids(), atoms.getResnums(), atoms.getIcodes(),
                    atoms.getResnames(), atoms.getNames()))

def generate_displacements(out):
    if not INPUT_PDB.is_file():
        raise FileNotFoundError(INPUT_PDB)
    if BETA_A <= 0 or not np.isfinite(BETA_A):
        raise ValueError("BETA_A 必须是有限正数")
    if MODE_NUMBER < 1 or MODE_SIGN not in (-1, 1):
        raise ValueError("MODE_NUMBER >= 1，MODE_SIGN 必须为 ±1")

    original = read_pdb(INPUT_PDB)
    selected = original.select(ATOM_SELECTION)
    if selected is None:
        raise ValueError(f"原子选择为空：{ATOM_SELECTION}")
    atoms = selected.copy()
    r0 = np.asarray(atoms.getCoords(), dtype=float)
    n = len(r0)
    if EXPECTED_ATOM_COUNT is not None and n != EXPECTED_ATOM_COUNT:
        raise ValueError(f"原子数 {n} != {EXPECTED_ATOM_COUNT}")

    anm = pd.ANM(f"{TASK_PREFIX}_all_atom")
    anm.buildHessian(atoms, cutoff=ANM_CUTOFF_A, gamma=ANM_GAMMA)
    # 分离的水/离子等可能使网络不连通，不能直接解释为完整分子的模态。
    k = anm.getKirchhoff()
    components = connected_components(k < 0, directed=False, return_labels=False)
    if components != 1:
        raise ValueError(f"ANM 网络有 {components} 个不连通部分；请检查原子选择或 cutoff")
    anm.calcModes(n_modes=MODE_NUMBER, zeros=False)
    if anm.numModes() < MODE_NUMBER:
        raise ValueError("可用非零模态不足")
    mode = anm[MODE_NUMBER - 1]
    u = np.asarray(mode.getEigvec(), dtype=float).reshape(n, 3)
    u /= np.linalg.norm(u)
    # 固定特征向量的任意正负号，使同一输入的差分符号可复现。
    if u.ravel()[np.argmax(np.abs(u))] < 0:
        u = -u
    u *= MODE_SIGN

    # 只用基准构象定义中心；正负构象绝不分别重新居中或旋转。
    center = (r0.min(axis=0) + r0.max(axis=0)) / 2
    coords = {}
    for label, sign in (("plus", 1), ("minus", -1)):
        moved = atoms.copy()
        moved.setCoords(r0 + sign * BETA_A * u)
        path = out / f"{label}.pdb"
        pd.writePDB(str(path), moved)
        # 按要求，从实际写出的 PDB 再读取坐标用于小球建模。
        reread = pd.parsePDB(str(path), model=1, altloc=PDB_ALTLOC)
        if reread is None or atom_ids(reread) != atom_ids(atoms):
            raise ValueError(f"{path.name} 写入/重读后的原子顺序发生变化")
        coords[label] = np.asarray(reread.getCoords(), dtype=float)

    actual_delta = coords["plus"] - coords["minus"]
    ideal_delta = 2 * BETA_A * u
    relative_error = np.linalg.norm(actual_delta - ideal_delta) / (2 * BETA_A)
    if np.linalg.norm(actual_delta) == 0:
        raise ValueError("PDB 舍入后正负构象完全相同，请增大 BETA_A")
    if relative_error > PDB_ERROR_WARNING_FRACTION:
        warnings.warn(f"PDB 舍入造成的扰动向量相对误差为 {relative_error:.2%}；"
                      "请增大 beta，或改用高精度坐标建模。")
    np.savez(out / "mode_and_coordinates.npz", reference_A=r0, unit_mode=u,
             ideal_plus_A=r0 + BETA_A*u, ideal_minus_A=r0 - BETA_A*u,
             pdb_plus_A=coords["plus"], pdb_minus_A=coords["minus"], center_A=center)
    metadata = {
        "input_pdb": str(INPUT_PDB.resolve()), "atom_selection": ATOM_SELECTION,
        "n_atoms": n, "mode_number": MODE_NUMBER, "mode_sign": MODE_SIGN,
        "eigenvalue": float(mode.getEigval()), "beta_A": BETA_A,
        "ideal_rms_displacement_A": BETA_A / np.sqrt(n),
        "ideal_pair_distance_A": 2 * BETA_A,
        "pdb_pair_distance_A": float(np.linalg.norm(actual_delta)),
        "pdb_pair_projection_on_mode_A": float(np.sum(actual_delta*u)),
        "pdb_rounding_relative_error": float(relative_error),
        "center_A": center.tolist(),
        "pdb_sha256": {label: hashlib.sha256((out / f"{label}.pdb").read_bytes()).hexdigest()
                       for label in ("plus", "minus")},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return {label: r - center for label, r in coords.items()}

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_displacements(OUTPUT_DIR)
    config = {k: v for k, v in globals().items() if k.isupper()}
    (OUTPUT_DIR / "generation_config.json").write_text(
        json.dumps(config, default=str, indent=2), encoding="utf-8")
    print(f"已生成 plus.pdb、minus.pdb 和 metadata.json：{OUTPUT_DIR.resolve()}")
    print("接下来运行 02_tidy3d_difference.py。")


if __name__ == "__main__":
    main()

