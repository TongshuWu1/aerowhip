"""Fit DDER EI and Cb for the one-node pivot model, then validate the winner."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experimental_data.cable_fit import fit_pivot_cable


def main() -> int:
    result = fit_pivot_cable()
    fitted = result["fitted_parameters"]
    training = result["training"]
    validation = result["validation"]
    print(
        f"Fitted EI={fitted['EI_n_m2']:.9g} N m^2, "
        f"Cb={fitted['Cb_n_m2_s']:.9g} N m^2 s"
    )
    print(
        "Marker RMSE improvement: "
        f"training {training['marker_rmse_improvement_percent']:.2f}%, "
        f"validation {validation['marker_rmse_improvement_percent']:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
