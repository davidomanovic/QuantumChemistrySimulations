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
from ccpy.drivers.driver import Driver
import io
import contextlib

start, stop, step = 0.7, 2.4, 0.1
bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)
molecule = "n2"
basis = "cc-pvtz"
atom = lambda R: [["N", (0.0, 0, 0)], ["N", (R, 0, 0)]]
n_f = 2
OUT_CSV = Path(f"output/{molecule}_{basis}.csv")


def append_row_csv(path, row_dict, header):
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        w.writerow(row_dict)


def main():
    if OUT_CSV.exists():
        OUT_CSV.unlink()

    header = [
        "R",
        "E_HF",
        "E_CCSD",
        "E_CCSD(T)",
        "E_CCSDT",
    ]
    print(",".join(header), flush=True)

    for R in bond_distance_range:
        mol = pyscf.gto.Mole()
        mol.build(atom=atom(R), basis=basis, symmetry="D2h", verbose=0)
        active_space = range(n_f, mol.nao_nr())

        scf = pyscf.scf.RHF(mol).run()
        norb = len(active_space)
        nelec = int(sum(scf.mo_occ[active_space]))
        nelec_alpha = (nelec + mol.spin) // 2
        nelec_beta = (nelec - mol.spin) // 2

        cas = pyscf.mcscf.RCASCI(scf, ncas=norb, nelecas=nelec)
        mo_coeff = cas.sort_mo(active_space, base=0)

        ccsd = pyscf.cc.RCCSD(
            scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
        )
        ccsd.kernel()
        e_t = ccsd.ccsd_t()

        driver = Driver.from_pyscf(scf, nfrozen=n_f)
        driver.options["energy_convergence"] = 1.0e-7
        driver.options["amp_convergence"] = 1.0e-7
        driver.options["maximum_iterations"] = 80

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            driver.run_cc(method="ccsdt")

        E_HF = scf.e_tot
        E_CCSD = ccsd.e_tot
        E_CCSD_T = E_CCSD + e_t
        E_CCSDT = driver.system.reference_energy + driver.correlation_energy

        mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)

        row = {
            "R": f"{R:.6f}",
            "E_HF": f"{E_HF:.12f}",
            "E_CCSD": f"{E_CCSD:.12f}",
            "E_CCSD(T)": f"{E_CCSD_T:.12f}",
            "E_CCSDT": f"{E_CCSDT:.12f}",
        }
        print(",".join([row[k] for k in header]), flush=True)
        append_row_csv(OUT_CSV, row, header)

    print(f"Wrote CSV: {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
