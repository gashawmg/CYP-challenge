#!/usr/bin/env python
r"""Verify an environment can actually run the tabular track. Exits non-zero if not.

    python scripts/check_env.py

Checks more than "does it import". Version numbers lie about whether a GPU kernel will
launch, so this runs a real matmul on the device and a real TabM forward/backward pass --
the exact module scripts/fit_tabular.py builds. It also parses a molecule, computes a
descriptor and a Bemis-Murcko scaffold, because an RDKit that imports but cannot sanitise
is a failure mode that only shows up an hour into a feature build.

Nothing here installs or changes anything. Safe to run in any env, including admet_gpu.
"""
from __future__ import annotations

import importlib
import platform
import sys

OK, WARN, FAIL = "PASS", "WARN", "FAIL"
results = []


def record(name, status, detail=""):
    results.append((name, status, detail))
    mark = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]
    print(f"[{mark}] {name:34s} {detail}")


# which DISTRIBUTION provides an import name. Matters for mordred, where the working
# package is `mordredcommunity` and the abandoned one is `mordred`, both importing as
# `mordred` -- the version string alone cannot tell them apart.
DISTS = {"mordred": ["mordredcommunity", "mordred"],
         "sklearn": ["scikit-learn"], "rdkit": ["rdkit", "rdkit-pypi"]}


def version_of(mod):
    name = mod.__name__
    try:
        from importlib import metadata
        for dist in DISTS.get(name, [name]):
            try:
                return f"{metadata.version(dist)}  [{dist}]"
            except metadata.PackageNotFoundError:
                continue
    except Exception:                                                # noqa: BLE001
        pass
    for attr in ("__version__", "version", "VERSION"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
    return "?"


def check_import(name, required, note=""):
    try:
        m = importlib.import_module(name)
    except Exception as e:                                           # noqa: BLE001
        record(name, FAIL if required else WARN,
               f"not importable ({type(e).__name__})" + (f" -- {note}" if note else ""))
        return None
    record(name, OK, version_of(m))
    return m


def main() -> int:
    print(f"python {platform.python_version()}  ({sys.executable})")
    print(f"platform {platform.platform()}\n")

    print("--- required ----------------------------------------------------------")
    np = check_import("numpy", True)
    pd = check_import("pandas", True)
    rd = check_import("rdkit", True)
    sk = check_import("sklearn", True, "needed by fit_tabular for hgb/rf/ridge/svr/--svd")

    print("\n--- rdkit actually works ----------------------------------------------")
    if rd is not None:
        try:
            from rdkit import Chem, RDLogger
            from rdkit.Chem import Descriptors
            from rdkit.Chem.Scaffolds import MurckoScaffold
            RDLogger.DisableLog("rdApp.*")
            m = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")       # aspirin
            assert m is not None, "could not parse aspirin"
            logp = Descriptors.MolLogP(m)
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
            n_desc = len(Descriptors.descList)
            record("rdkit parse/descriptor/scaffold", OK,
                   f"{n_desc} descriptors, logP {logp:.2f}, scaffold '{sc}'")
        except Exception as e:                                       # noqa: BLE001
            record("rdkit parse/descriptor/scaffold", FAIL, f"{type(e).__name__}: {e}")

    print("\n--- torch and the GPU --------------------------------------------------")
    torch = check_import("torch", False, "needed only for --model tabm")
    dev = "cpu"
    if torch is not None:
        try:
            built = getattr(torch.version, "cuda", None)
            if not built:
                # WARN not FAIL: hgb/rf/ridge/svr are CPU models and run fine. Only
                # --model tabm wants the GPU, and it still works, just slowly.
                record("torch CUDA build", WARN,
                       "CPU-ONLY wheel. For GPU, reinstall with "
                       "--index-url https://download.pytorch.org/whl/cu124")
            elif not torch.cuda.is_available():
                record("torch CUDA build", WARN,
                       f"built for CUDA {built} but no device visible -- driver too old "
                       f"or no GPU; check nvidia-smi")
            else:
                dev = "cuda"
                i = torch.cuda.current_device()
                cap = torch.cuda.get_device_capability(i)
                name = torch.cuda.get_device_name(i)
                vram = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
                record("torch CUDA build", OK, f"runtime {built}, device '{name}'")
                sm = f"sm_{cap[0]}{cap[1]}"
                arch = torch.cuda.get_arch_list()
                if sm in arch:
                    record("GPU architecture supported", OK,
                           f"{sm} is in this wheel's kernel list, {vram:.1f} GiB VRAM")
                else:
                    record("GPU architecture supported", FAIL,
                           f"{sm} NOT in {arch} -- this wheel has no kernels for this "
                           f"card; pick a different CUDA tag")
        except Exception as e:                                       # noqa: BLE001
            record("torch CUDA build", FAIL, f"{type(e).__name__}: {e}")

        # a real kernel launch, not a capability string
        try:
            a = torch.randn(2048, 2048, device=dev)
            got = float((a @ a.T).diagonal().sum())
            assert got == got, "matmul produced NaN"
            record(f"matmul on {dev}", OK, f"2048x2048 ran, trace {got:,.0f}")
        except Exception as e:                                       # noqa: BLE001
            record(f"matmul on {dev}", FAIL, f"{type(e).__name__}: {e}")

        # the exact model fit_tabular builds: forward, backward, optimiser step
        try:
            from torch import nn
            k, width, d_in = 8, 256, 64

            class EnsLinear(nn.Module):
                def __init__(self, di, do):
                    super().__init__()
                    self.W = nn.Linear(di, do, bias=False)
                    self.r = nn.Parameter(torch.ones(k, di))
                    self.s = nn.Parameter(torch.ones(k, do))
                    self.b = nn.Parameter(torch.zeros(k, do))

                def forward(self, x):
                    return self.W(x * self.r) * self.s + self.b

            l1, head = EnsLinear(d_in, width).to(dev), EnsLinear(width, 1).to(dev)
            opt = torch.optim.AdamW(list(l1.parameters()) + list(head.parameters()),
                                    lr=1e-3)
            x = torch.randn(128, d_in, device=dev)
            y = torch.randn(128, device=dev)
            h = x.unsqueeze(1).expand(-1, k, -1)
            pred = head(torch.relu(l1(h))).squeeze(-1)
            loss = nn.HuberLoss(delta=0.5)(pred, y.unsqueeze(1).expand_as(pred))
            loss.backward(); opt.step()
            record(f"TabM fwd/bwd/step on {dev}", OK,
                   f"k={k}, loss {loss.detach().item():.4f}")
        except Exception as e:                                       # noqa: BLE001
            record(f"TabM fwd/bwd/step on {dev}", FAIL, f"{type(e).__name__}: {e}")

        if dev == "cuda":
            try:
                torch.set_float32_matmul_precision("high")
                record("TF32 matmul precision", OK,
                       "set to 'high' -- faster on Ampere, used by train_cyp_clean.py")
            except Exception as e:                                   # noqa: BLE001
                record("TF32 matmul precision", WARN, str(e))

    print("\n--- optional feature blocks --------------------------------------------")
    for mod, what in [("mordred", "1,613 Mordred descriptors (pip install "
                                  "mordredcommunity -- NOT mordred)"),
                      ("transformers", "frozen ChemBERTa embeddings"),
                      ("unimol_tools", "frozen Uni-Mol embeddings"),
                      ("tabpfn", "TabPFN v2 tabular foundation model")]:
        try:
            m = importlib.import_module(mod)
            record(mod, OK, version_of(m))
        except Exception:                                            # noqa: BLE001
            record(mod, WARN, f"absent -- {what}")

    print("\n--- chemprop track (separate env; expected absent here) -----------------")
    for mod in ("chemprop", "lightning"):
        try:
            m = importlib.import_module(mod)
            record(mod, OK, version_of(m))
        except Exception:                                            # noqa: BLE001
            record(mod, WARN, "absent -- train_cyp_clean.py runs in admet_gpu instead")

    # ---------------------------------------------------------------- verdict
    fails = [r for r in results if r[1] == FAIL]
    warns = [r for r in results if r[1] == WARN]
    print("\n" + "=" * 74)
    if fails:
        print(f"  {len(fails)} FAILURE(S) -- the tabular track will not run:")
        for n, _, d in fails:
            print(f"    {n}: {d}")
        return 1
    gpu = any(n.endswith("on cuda") for n, s, _ in results if s == OK)
    if gpu:
        print("  READY, WITH GPU. --model tabm will be fast.")
    else:
        print("  READY, CPU ONLY. hgb / rf / ridge / svr run fine; --model tabm will\n"
              "  work but slowly. If you expected a GPU on this machine, that is the\n"
              "  one thing to fix before building features: check nvidia-smi, then\n"
              "  match the CUDA tag of the torch wheel to the driver.")
    print(f"  {len(warns)} optional component(s) absent; everything required is present.")
    print("\n  next:  python scripts/build_features.py --probe")
    print("         python scripts/eval_service.py init")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
