import time
import csv
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

start, stop, step = 1.1, 1.1, 0.05
bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)
molecule = "n2(sq)"
basis = "sto-6g"
atom = lambda R: [["N", (0.0, 0, 0)], ["N", (R, 0, 0)]]
n_f = 2

OUT_CSV = Path(f"output/{molecule}_{basis}.csv")
TRACE_CSV = Path(f"output/{molecule}_{basis}_trace.csv")


def append_row_csv(path, row_dict, header):
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        w.writerow(row_dict)


def main():
    if OUT_CSV.exists():
        OUT_CSV.unlink()
    if TRACE_CSV.exists():
        TRACE_CSV.unlink()

    x_prev1 = None
    n_reps = 7
    trace_header = ["R", "iter", "nfev", "energy", "error", "method"]

    for R in bond_distance_range:
        mol = pyscf.gto.Mole()
        mol.build(atom=atom(R), basis=basis, symmetry="Dooh", max_memory=256 * 1000, verbose=0)
        active_space = range(n_f, mol.nao_nr())

        scf = pyscf.scf.RHF(mol).run()
        norb = len(active_space)
        nelec = int(sum(scf.mo_occ[active_space]))
        nelec_alpha = (nelec + mol.spin) // 2
        nelec_beta = (nelec - mol.spin) // 2

        cas = pyscf.mcscf.RCASCI(scf, ncas=norb, nelecas=nelec)
        mo_coeff = cas.sort_mo(active_space, base=0)

        ccsd = pyscf.cc.RCCSD(scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space])
        ccsd.kernel()

        E_HF = scf.e_tot

        cas.fix_spin_(ss=0)
        cas.kernel(mo_coeff=mo_coeff)
        E_FCI = cas.e_tot

        mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
        norb = mol_data.norb
        nelec = mol_data.nelec
        H = ffsim.linear_operator(mol_data.hamiltonian, norb=norb, nelec=nelec)
        Phi0 = ffsim.hartree_fock_state(norb, nelec)

        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        pairs_ab = [(p, p) for p in range(norb)]
        lucj1_seed = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=ccsd.t2,
            t1=ccsd.t1,
            n_reps=n_reps,
            interaction_pairs=(pairs_aa, pairs_ab),
            optimize=True
        )
        psi1 = ffsim.apply_unitary(Phi0, lucj1_seed, norb=norb, nelec=nelec)
        E_LuCJ = float(np.vdot(psi1, H @ psi1).real)

        def params_to_vec(x: np.ndarray) -> np.ndarray:
            operator = ffsim.UCJOpSpinBalanced.from_parameters(
                x,
                norb=norb,
                n_reps=n_reps,
                interaction_pairs=(pairs_aa, pairs_ab),
                with_final_orbital_rotation=True,
            )
            return ffsim.apply_unitary(Phi0, operator, norb=norb, nelec=nelec)

        def energy_from_params(x: np.ndarray) -> float:
            vec = params_to_vec(x)
            return float(np.vdot(vec, H @ vec).real)

        if x_prev1 is None:
            x01 = lucj1_seed.to_parameters(interaction_pairs=(pairs_aa, pairs_ab))
        else:
            x01 = x_prev1

        print(f"R = {R:.3f}")
        print("Params:", len(x01))
        print("LM iter,nfev,E,error")

        def lm_callback(res):
            it = res.nit
            e = res.fun
            err = abs(E_FCI - e)
            append_row_csv(
                TRACE_CSV,
                {
                    "R": f"{R:.6f}",
                    "iter": it,
                    "nfev": res.nfev,
                    "energy": f"{e:.12f}",
                    "error": f"{err:.12f}",
                    "method": "LM",
                },
                trace_header,
            )
            print(f"{it:4d},{res.nfev:5d},{e:.12f},{err:.12f}", flush=True)

        res1 = minimize_linear_method(
            params_to_vec,
            H,
            x0=x01,
            maxiter=20,
            callback=lm_callback,
            optimize_regularization=True,
            optimize_variation=True,
            ftol=1e-12,
            gtol=1e-12,
        )
        E_LM = float(res1.fun)

        class EnergyFunctor:
            def __init__(self):
                self.nfev = 0
                self.last_val = None

            def __call__(self, x: np.ndarray) -> float:
                e = energy_from_params(x)
                self.nfev += 1
                self.last_val = e
                return e

        print("COBYLA iter,nfev,E,error")
        cobyla_fun = EnergyFunctor()

        def cobyla_callback(xk):
            cobyla_callback.iter += 1
            e = energy_from_params(xk)
            err = abs(E_FCI - e)
            append_row_csv(
                TRACE_CSV,
                {
                    "R": f"{R:.6f}",
                    "iter": cobyla_callback.iter,
                    "nfev": cobyla_fun.nfev,
                    "energy": f"{e:.12f}",
                    "error": f"{err:.12f}",
                    "method": "COBYLA",
                },
                trace_header,
            )
            print(f"{cobyla_callback.iter:4d},{cobyla_fun.nfev:5d},{e:.12f},{err:.12f}", flush=True)

        cobyla_callback.iter = 0

        res2 = scipy.optimize.minimize(
            cobyla_fun,
            x01,
            method="COBYLA",
            callback=cobyla_callback,
            options={"maxiter": 25000},
            tol=1e-12,
        )
        E_COBYLA = float(res2.fun)

        print("L-BFGS-B iter,nfev,E,error")
        lbfgs_fun = EnergyFunctor()

        def lbfgsb_callback(xk):
            lbfgsb_callback.iter += 1
            e = energy_from_params(xk)
            err = abs(E_FCI - e)
            append_row_csv(
                TRACE_CSV,
                {
                    "R": f"{R:.6f}",
                    "iter": lbfgsb_callback.iter,
                    "nfev": lbfgs_fun.nfev,
                    "energy": f"{e:.12f}",
                    "error": f"{err:.12f}",
                    "method": "L-BFGS-B",
                },
                trace_header,
            )
            print(f"{lbfgsb_callback.iter:4d},{lbfgs_fun.nfev:5d},{e:.12f},{err:.12f}", flush=True)

        lbfgsb_callback.iter = 0

        res3 = scipy.optimize.minimize(
            lbfgs_fun,
            x01,
            method="L-BFGS-B",
            callback=lbfgsb_callback,
            options={"maxiter": 50, "ftol": 1e-18, "gtol": 1e-18, "maxfun": 25000},
        )
        E_LBFGSB = float(res3.fun)

        print("Final energies:")
        print(f"  FCI      {E_FCI:.12f}")
        print(f"  LM       {E_LM:.12f}")
        print(f"  COBYLA   {E_COBYLA:.12f}")
        print(f"  L-BFGS-B {E_LBFGSB:.12f}")

        x_prev1 = res3.x

    print(f"Wrote trace CSV:   {TRACE_CSV.resolve()}")


if __name__ == "__main__":
    main()
