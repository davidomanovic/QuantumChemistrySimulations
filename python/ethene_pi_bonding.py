import csv
import os
from pathlib import Path
import scipy.optimize
import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import pyscf.cc
import ffsim
from ffsim.optimize import minimize_linear_method

start, stop, step = 0.9, 2.8, 0.05
bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)
n_f = 0
molecule = "c2h4"
basis = "cc-pvtz"
OUT_CSV = Path(f"output/{molecule}_{basis}.csv")
pyscf.lib.num_threads(48)

def append_row_csv(path, row_dict, header):
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        w.writerow(row_dict)

def VQE(H, Phi0, norb, nelec, pairs_aa, pairs_ab, n_reps, x0):
    def vec_from_params(x):
        op = ffsim.UCJOpSpinBalanced.from_parameters(
            x,
            norb=norb,
            n_reps=n_reps,
            interaction_pairs=(pairs_aa, pairs_ab),
            with_final_orbital_rotation=True,
        )
        return ffsim.apply_unitary(Phi0, op, norb=norb, nelec=nelec)

    res = minimize_linear_method(vec_from_params, H, x0=x0, maxiter=10)
    energy = float(res.fun)
    return energy, res.x

def main():
    if OUT_CSV.exists():
        OUT_CSV.unlink()

    header = [
        "R",
        "E_FCI",
        "E_HF",
        "E_CCSD",
        "E_LuCJ",
        "E_LuCJ1(VQE)",
        "E_LuCJ2(VQE)",
        "E_LuCJ4(VQE)",
    ]
    print(",".join(header), flush=True)

    x_prev1 = None
    x_prev2 = None
    x_prev3 = None

    for R in bond_distance_range:
        a = 0.5 * R
        b = a + 0.5626
        c = 0.9289
        mol = pyscf.gto.Mole()
        mol.build(
            atom=[
                ["C", (0, 0, a)],
                ["C", (0, 0, -a)],
                ["H", (0, c, b)],
                ["H", (0, -c, b)],
                ["H", (0, c, -b)],
                ["H", (0, -c, -b)],
            ],
            basis=basis,
            symmetry="d2h",
            verbose=0,
        )

        scf = pyscf.scf.RHF(mol).run()
        active_space = range(mol.nelectron // 2 - 2, mol.nelectron // 2 + 2) # pi-bonding active space

        mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
        norb = mol_data.norb
        nelec = mol_data.nelec

        cas = pyscf.mcscf.RCASCI(scf, ncas=norb, nelecas=nelec)
        mo_coeff = cas.sort_mo(active_space, base=0)
        cas.kernel(mo_coeff=mo_coeff)
        E_FCI = cas.e_tot

        norb = len(active_space)
        nelec = int(sum(scf.mo_occ[active_space]))
        nelec_alpha = (nelec + mol.spin) // 2
        nelec_beta = (nelec - mol.spin) // 2

        ccsd = pyscf.cc.RCCSD(
            scf,
            frozen=[i for i in range(mol.nao_nr()) if i not in active_space],
        )
        ccsd.kernel()
        E_CCSD = ccsd.e_tot
        E_HF = scf.e_tot

        mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
        norb = mol_data.norb
        nelec = mol_data.nelec
        H = ffsim.linear_operator(mol_data.hamiltonian, norb=norb, nelec=nelec)
        Phi0 = ffsim.hartree_fock_state(norb, nelec)

        # heavy hex connectivity
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        pairs_ab = [(p, p) for p in range(0, norb, 4)]

        lucj1_seed = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=ccsd.t2,
            t1=ccsd.t1,
            n_reps=1,
            interaction_pairs=(pairs_aa, pairs_ab),
            optimize=True
        )
        psi1 = ffsim.apply_unitary(Phi0, lucj1_seed, norb=norb, nelec=nelec)
        E_LuCJ = float(np.vdot(psi1, H @ psi1).real)

        # Warm start from previous geometries (imporves convergence along dissociation where ccsd breaks down)
        if x_prev1 is None:
            x01 = lucj1_seed.to_parameters(interaction_pairs=(pairs_aa, pairs_ab))
        else:
            x01 = x_prev1

        E_LuCJ1_var, x_prev1 = VQE(
            H=H,
            Phi0=Phi0,
            norb=norb,
            nelec=nelec,
            pairs_aa=pairs_aa,
            pairs_ab=pairs_ab,
            n_reps=1,
            x0=x01,
        )

        lucj2_seed = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=ccsd.t2,
            t1=ccsd.t1,
            n_reps=2,
            interaction_pairs=(pairs_aa, pairs_ab),
            optimize=True
        )
        if x_prev2 is None:
            x02 = lucj2_seed.to_parameters(interaction_pairs=(pairs_aa, pairs_ab))
        else:
            x02 = x_prev2

        E_LuCJ2_var, x_prev2 = VQE(
            H=H,
            Phi0=Phi0,
            norb=norb,
            nelec=nelec,
            pairs_aa=pairs_aa,
            pairs_ab=pairs_ab,
            n_reps=2,
            x0=x02,
        )

        lucj3_seed = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=ccsd.t2,
            t1=ccsd.t1,
            n_reps=4,
            interaction_pairs=(pairs_aa, pairs_ab),
            optimize=True
        )
        if x_prev3 is None:
            x03 = lucj3_seed.to_parameters(interaction_pairs=(pairs_aa, pairs_ab))
        else:
            x03 = x_prev3

        E_LuCJ3_var, x_prev3 = VQE(
            H=H,
            Phi0=Phi0,
            norb=norb,
            nelec=nelec,
            pairs_aa=pairs_aa,
            pairs_ab=pairs_ab,
            n_reps=4,
            x0=x03,
        )

        row = {
            "R": f"{R:.6f}",
            "E_FCI": f"{E_FCI:.12f}",
            "E_HF": f"{E_HF:.12f}",
            "E_CCSD": f"{E_CCSD:.12f}",
            "E_LuCJ": f"{E_LuCJ:.12f}",
            "E_LuCJ1(VQE)": f"{E_LuCJ1_var:.12f}",
            "E_LuCJ2(VQE)": f"{E_LuCJ2_var:.12f}",
            "E_LuCJ4(VQE)": f"{E_LuCJ3_var:.12f}",
        }
        print(",".join(row[k] for k in header), flush=True)
        append_row_csv(OUT_CSV, row, header)

    print(f"Wrote CSV: {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()
