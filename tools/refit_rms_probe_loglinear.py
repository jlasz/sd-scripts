"""Reproduce the loglinear_v1 RMS-probe constants.

Offline maintenance tool. It reuses the reviewed calibration table of
refit_rms_probe_calibration.py and adds the completed runs that are not part of
the piecewise calibration. The model is one unweighted least-squares fit:

    log(final_rms) = intercept
                     + a * log(total_steps)
                     + b * log(batches_per_epoch / gradient_accumulation)
                     + c * log(reference_rms)

To add a run, append a CalibrationRow to EXTRA_ROWS and run this file; copy the
printed constants into library/rms_step_probe.py.
"""

from __future__ import annotations

import math

import numpy as np

from refit_rms_probe_calibration import ROWS as PIECEWISE_ROWS
from refit_rms_probe_calibration import CalibrationRow


# Same-GA step-500 views of completed production curves that the piecewise
# table does not contain. early_energy_slope is recorded but not used here.
EXTRA_ROWS = (
    # Third 001 revision: 109 batches per epoch, GA 6, 4,600 steps.
    CalibrationRow("object_character_001", 109, 3.9533202e-5, 6, 2.0512864155, 4600, 9.0057752e-5),
    # Ahsoka: 258-, 243-, and 210-image revisions at GA 4.
    CalibrationRow("ahsoka", 258, 3.2891719e-5, 4, 1.9514609796, 7200, 9.0968387e-5),
    CalibrationRow("ahsoka", 243, 3.3271486e-5, 4, 1.9235098361, 6400, 8.3515770e-5),
    CalibrationRow("ahsoka", 210, 3.3617045e-5, 4, 2.0263120949, 6400, 8.8331746e-5),
)
ROWS = tuple(PIECEWISE_ROWS) + EXTRA_ROWS


def features(rows: tuple[CalibrationRow, ...]) -> np.ndarray:
    return np.asarray(
        [
            (
                1.0,
                math.log(row.total_steps),
                math.log(row.batches_per_epoch / row.production_ga),
                math.log(row.reference_rms),
            )
            for row in rows
        ],
        dtype=np.float64,
    )


def targets(rows: tuple[CalibrationRow, ...]) -> np.ndarray:
    return np.log(np.asarray([row.final_rms for row in rows], dtype=np.float64))


def refit(rows: tuple[CalibrationRow, ...] = ROWS) -> np.ndarray:
    return np.linalg.lstsq(features(rows), targets(rows), rcond=None)[0]


def relative_errors(coefficients: np.ndarray, rows: tuple[CalibrationRow, ...]) -> np.ndarray:
    return np.exp(features(rows) @ coefficients - targets(rows)) - 1.0


def leave_one_family_out(rows: tuple[CalibrationRow, ...] = ROWS) -> np.ndarray:
    errors = []
    for family in sorted({row.family for row in rows}):
        training_rows = tuple(row for row in rows if row.family != family)
        held_out_rows = tuple(row for row in rows if row.family == family)
        errors.extend(relative_errors(refit(training_rows), held_out_rows))
    return np.asarray(errors)


def main() -> None:
    coefficients = refit()
    print("rows:", len(ROWS))
    print("LOGLINEAR_INTERCEPT =", repr(float(coefficients[0])))
    print("LOGLINEAR_STEPS_EXPONENT =", repr(float(coefficients[1])))
    print("LOGLINEAR_STEPS_PER_EPOCH_EXPONENT =", repr(float(coefficients[2])))
    print("LOGLINEAR_REFERENCE_RMS_EXPONENT =", repr(float(coefficients[3])))
    in_sample = np.abs(relative_errors(coefficients, ROWS))
    print(f"in-sample endpoint MAPE: {in_sample.mean() * 100:.3f}% (max {in_sample.max() * 100:.3f}%)")
    held_out = np.abs(leave_one_family_out())
    print(
        "leave-one-family-out endpoint MAPE: "
        f"{held_out.mean() * 100:.3f}% (max {held_out.max() * 100:.3f}%)"
    )
    unseen = relative_errors(refit(tuple(PIECEWISE_ROWS)), EXTRA_ROWS)
    print(
        "fit on the piecewise table only, endpoint error on EXTRA_ROWS:",
        ", ".join(f"{error * 100:+.2f}%" for error in unseen),
    )


if __name__ == "__main__":
    main()
