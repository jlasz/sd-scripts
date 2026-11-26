"""Reproduce the Anima RMS-probe compressed-stage calibration constants.

This is an offline maintenance tool. It intentionally keeps the archived
operational probe inputs and final adapter RMS values in one reviewable table
instead of making runtime prediction depend on experiment directories.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


RIDGE_ALPHA = 0.001
PRIOR_STAGE_FACTORS = np.asarray((0.95263438, 0.98385027, 1.03754052, 1.02597483))
CROSSBREED_STAGE_SHAPES = np.asarray(
    (
        (0.910228518419122, 1.0216218853388563, 1.0236136216221736, 1.0445359746198482),
        (0.9415204401573268, 1.0091855732575818, 1.0199314151937091, 1.0293625713913828),
    )
)
CROSSBREED_STAGE_SHAPE = np.mean(CROSSBREED_STAGE_SHAPES, axis=0)
MARCIA_STAGE_VELOCITIES = np.asarray(
    (
        # 1,550-step and 2,400-step completed GA-5 schedules, followed by
        # the 1,700-step GA-11, 2,000-step GA-9, and 1,600-step GA-14
        # schedules.
        (2.400155e-8, 2.210202e-8, 2.322872e-8, 2.293994e-8),
        (
            2.312615460080274e-8,
            2.232675674906793e-8,
            2.4164681436872227e-8,
            2.6087100878601762e-8,
        ),
        (
            3.253366551635898e-8,
            3.419089968220384e-8,
            3.5470268224290684e-8,
            4.003662444815488e-8,
        ),
        (
            2.772597757652157e-8,
            2.9396111418272515e-8,
            3.026081558026929e-8,
            3.3129234859523805e-8,
        ),
        (
            3.2393149642857146e-8,
            3.2821576428571425e-8,
            3.62915244642857e-8,
            3.921755392857143e-8,
        ),
    )
)
MARCIA_STAGE_SHAPES = MARCIA_STAGE_VELOCITIES / np.mean(
    MARCIA_STAGE_VELOCITIES,
    axis=1,
    keepdims=True,
)
MARCIA_STAGE_SHAPE = np.mean(MARCIA_STAGE_SHAPES, axis=0)
BEASTGIRL_STAGE_SHAPES = np.asarray(
    (
        (
            0.9751315362180831,
            0.9268944353291072,
            1.0442107351508483,
            1.0537632933019614,
        ),
        (
            0.860464886024853,
            0.9740218504766431,
            1.0525559010097465,
            1.1129573624887577,
        ),
        (
            0.880320904149621,
            0.9925945694574024,
            1.0112713618724714,
            1.1158131645205052,
        ),
    )
)
BEASTGIRL_STAGE_SHAPE = np.mean(BEASTGIRL_STAGE_SHAPES, axis=0)
# Completed GA-6 001 production curves: 4,200 steps with 122 batches per
# epoch, then 4,600 steps with the revised 119-batch dataset.
OBJECT_CHARACTER_001_STAGE_SHAPES = np.asarray(
    (
        (
            0.9349471823737898,
            1.0010818367081487,
            1.0125576404927095,
            1.0514133404253514,
        ),
        (
            0.955489716862284,
            0.9810965097925542,
            1.017635837089424,
            1.0457779362557378,
        ),
    )
)
OBJECT_CHARACTER_001_STAGE_SHAPE = np.mean(
    OBJECT_CHARACTER_001_STAGE_SHAPES,
    axis=0,
)
STAGE_FACTORS = tuple(
    (
        6.0 * PRIOR_STAGE_FACTORS
        + CROSSBREED_STAGE_SHAPE
        + MARCIA_STAGE_SHAPE
        + BEASTGIRL_STAGE_SHAPE
        + OBJECT_CHARACTER_001_STAGE_SHAPE
    )
    / 10.0
)

# Per-family means of retained_energy_mean. M'rissi predates this telemetry.
RETENTION_FAMILY_MEANS = {
    "izutsumi": np.asarray((0.921853, 0.91098125, 0.89459475, 0.86145975)),
    "neeko": np.asarray(
        (0.9259206666666667, 0.9147833333333334, 0.8976413333333334, 0.8647483333333333)
    ),
    "wilykit": np.asarray((0.9310895, 0.9207085, 0.903313, 0.8708695)),
    "mutio": np.asarray((0.929357, 0.917604, 0.900059, 0.864986)),
    "rosine": np.asarray((0.9395396, 0.9296544, 0.9127912, 0.8807426)),
    "crossbreed_priscilla": np.asarray(
        (
            np.mean((0.9301674962043762, 0.9325389266014099)),
            np.mean((0.9186212420463562, 0.9203118085861206)),
            np.mean((0.9011589288711548, 0.9023491144180298)),
            np.mean((0.8675388693809509, 0.8682758212089539)),
        )
    ),
    "marcia": np.asarray(
        (
            np.mean(
                (
                    0.9397159218788147,
                    0.9446578025817871,
                    0.9477311372756958,
                    0.9459108114242554,
                    0.947370,
                )
            ),
            np.mean(
                (
                    0.931081235408783,
                    0.9350574016571045,
                    0.938911497592926,
                    0.9370183348655701,
                    0.938918,
                )
            ),
            np.mean(
                (
                    0.9144988656044006,
                    0.9179385900497437,
                    0.9211618900299072,
                    0.9195295572280884,
                    0.921458,
                )
            ),
            np.mean(
                (
                    0.8838304877281189,
                    0.8856008052825928,
                    0.8903316855430603,
                    0.8870620727539062,
                    0.891531,
                )
            ),
        )
    ),
    "beastgirl": np.asarray(
        (
            np.mean((0.9472075700759888, 0.953414797782898, 0.9501983523368835)),
            np.mean((0.9393881559371948, 0.9455373883247375, 0.9429237842559814)),
            np.mean((0.924223780632019, 0.9300624132156372, 0.927065372467041)),
            np.mean((0.8950448036193848, 0.8998176455497742, 0.8984496593475342)),
        )
    ),
    "object_character_001": np.asarray(
        (
            np.mean((0.9328334331512451, 0.932612)),
            np.mean((0.9219856262207031, 0.920868)),
            np.mean((0.9042553901672363, 0.902960)),
            np.mean((0.8709542155265808, 0.868698)),
        )
    ),
}
SQUEEZE_RETENTION = tuple(
    (
        RETENTION_FAMILY_MEANS["izutsumi"]
        + RETENTION_FAMILY_MEANS["neeko"]
        + RETENTION_FAMILY_MEANS["wilykit"]
        + RETENTION_FAMILY_MEANS["mutio"]
        + 2.0 * RETENTION_FAMILY_MEANS["rosine"]
        + RETENTION_FAMILY_MEANS["crossbreed_priscilla"]
        + RETENTION_FAMILY_MEANS["marcia"]
        + RETENTION_FAMILY_MEANS["beastgirl"]
        + RETENTION_FAMILY_MEANS["object_character_001"]
    )
    / 10.0
)
FAMILY_SHARES = {
    "mrissi": 1.0,
    "izutsumi": 1.0,
    "neeko": 1.0,
    "wilykit": 1.0,
    "mutio": 1.0,
    "rosine": 2.0,
    "crossbreed_priscilla": 1.0,
    "marcia": 1.0,
    "beastgirl": 1.0,
    "object_character_001": 1.0,
}


@dataclass(frozen=True)
class CalibrationRow:
    family: str
    batches_per_epoch: int
    reference_rms: float
    production_ga: int
    early_energy_slope: float
    total_steps: int
    final_rms: float


ROWS = (
    CalibrationRow("mrissi", 132, 3.878279312630184e-5, 6, 2.1028907100460046, 4000, 8.425384599385171e-5),
    CalibrationRow("izutsumi", 250, 3.5439617931842804e-5, 6, 2.038203269953988, 4400, 6.785388119627348e-5),
    CalibrationRow("izutsumi", 250, 3.469298826530576e-5, 5, 2.006614663143924, 5460, 7.746604988565678e-5),
    CalibrationRow("izutsumi", 252, 3.3941563742700964e-5, 4, 1.945756760325379, 6000, 7.868044894115675e-5),
    CalibrationRow("izutsumi", 262, 3.417661355342716e-5, 4, 2.018023225383142, 6400, 8.171637791552906e-5),
    CalibrationRow("neeko", 193, 3.626849939802868e-5, 6, 2.0599586690458014, 4275, 7.425300935910043e-5),
    CalibrationRow("neeko", 193, 3.633632877608761e-5, 5, 2.0715826342080725, 4800, 7.806661638101423e-5),
    CalibrationRow("neeko", 203, 3.568904139683582e-5, 5, 1.9881853670471428, 5200, 8.129191504544343e-5),
    CalibrationRow("wilykit", 157, 3.804286006720449e-5, 5, 2.063401097961805, 4600, 7.737921127199928e-5),
    CalibrationRow("wilykit", 157, 3.804286006720449e-5, 5, 2.063401097961805, 5200, 8.593323702724446e-5),
    CalibrationRow("mutio", 172, 3.6507251192282416e-5, 5, 1.996520566790334, 5400, 8.701831151862569e-5),
    CalibrationRow("rosine", 99, 4.469153373812116e-5, 8, 2.1870861288967656, 3300, 8.300249220144348e-5),
    CalibrationRow("rosine", 95, 4.4754658783445186e-5, 9, 2.216775240715489, 3000, 8.007592923876682e-5),
    CalibrationRow("rosine", 104, 4.29566212128835e-5, 7, 2.139091157515416, 3500, 8.277126014895049e-5),
    CalibrationRow("rosine", 103, 3.648383790277876e-5, 2, 2.085742066127855, 4000, 6.372531148731839e-5),
    CalibrationRow("rosine", 110, 4.176764188992321e-5, 6, 2.1384678815539027, 4000, 8.552275888033521e-5),
    CalibrationRow("rosine", 117, 3.7124011e-5, 2, 2.01871264275116, 6900, 9.4962474e-5),
    CalibrationRow("rosine", 112, 4.4364149e-5, 9, 2.1487726165980137, 3400, 8.639951556688175e-5),
    CalibrationRow("rosine", 119, 4.293570495580571e-5, 6, 2.0936642515103694, 4200, 8.993679512059316e-5),
    CalibrationRow("rosine", 111, 4.5584457060263114e-5, 7, 2.1592672694926716, 3600, 8.496370015504847e-5),
    CalibrationRow("rosine", 103, 4.6760884669408374e-5, 10, 2.166961419148628, 2900, 7.965448157549462e-5),
    # Probe 1 was GA 6. These are its production-conditioned GA 7 values,
    # matching the values presented to the runtime compressed-rate regression.
    CalibrationRow("crossbreed_priscilla", 93, 3.858837416025104e-5, 7, 2.126722772584375, 3300, 7.7368463526e-5),
    # The second run probed and trained at GA 7, so no GA transfer is needed.
    CalibrationRow("crossbreed_priscilla", 93, 3.842611053225904e-5, 7, 2.0053641727895593, 3700, 8.623025450275907e-5),
    # Same-GA schedule-aware Probe-1 views from the completed 1,550-, 2,400-,
    # 1,700-, 2,000-, and 1,600-step Marcia trajectories. They split one
    # family share.
    CalibrationRow("marcia", 22, 4.7060277489e-5, 5, 2.1234049812, 1550, 5.9780546789e-5),
    CalibrationRow("marcia", 22, 5.0705232640351776e-5, 5, 2.1984010211552754, 2400, 8.776551913866043e-5),
    CalibrationRow("marcia", 22, 6.54549402987649e-5, 11, 2.306234594209267, 1700, 9.212609184623723e-5),
    CalibrationRow("marcia", 27, 5.4083582557090604e-5, 9, 2.17195124877972, 2000, 9.194107419552754e-5),
    CalibrationRow("marcia", 28, 6.351642683420896e-5, 14, 2.1520303378229664, 1600, 8.818310969338425e-5),
    # Same-GA schedule-aware views of the completed Beastgirl trajectories.
    CalibrationRow(
        "beastgirl",
        15,
        5.019000006813414e-5,
        4,
        2.1833027981860034,
        2200,
        8.158656763806009e-5,
    ),
    CalibrationRow(
        "beastgirl",
        22,
        6.370470275181598e-5,
        11,
        2.2177239901442296,
        1600,
        8.717409915077944e-5,
    ),
    CalibrationRow(
        "beastgirl",
        24,
        6.72222426902599e-5,
        12,
        2.2565366390239823,
        1450,
        7.998233372321603e-5,
    ),
    CalibrationRow(
        # The two 001 revisions split one object-and-character family share.
        # Tkl is intentionally excluded from every calibration aggregate.
        "object_character_001",
        122,
        3.8101557082479796e-5,
        6,
        2.0491012957960426,
        4200,
        7.982355725304759e-5,
    ),
    CalibrationRow(
        "object_character_001",
        119,
        3.8070918e-5,
        6,
        2.0700419993815973,
        4600,
        8.69400831157966e-5,
    ),
)


def segment_lengths(total_steps: int) -> tuple[int, ...]:
    boundaries = (0, *(total_steps * index // 5 for index in range(1, 5)), total_steps)
    return tuple(end - start for start, end in zip(boundaries, boundaries[1:]))


def predict(row: CalibrationRow, velocity: float) -> float:
    lengths = segment_lengths(row.total_steps)
    rms = row.reference_rms * math.sqrt(
        1.0 + row.early_energy_slope * (lengths[0] - 500) / 1000.0
    )
    for factor, retention, length in zip(STAGE_FACTORS, SQUEEZE_RETENTION, lengths[1:]):
        rms *= math.sqrt(retention)
        rms += row.reference_rms * velocity * factor * length / 1000.0
    return rms


def endpoint_velocity(row: CalibrationRow) -> float:
    zero = predict(row, 0.0)
    unit_gain = predict(row, 1.0) - zero
    return (row.final_rms - zero) / unit_gain


def features(row: CalibrationRow) -> np.ndarray:
    return np.asarray(
        (
            math.log(row.batches_per_epoch),
            math.log(row.reference_rms),
            math.log(row.production_ga),
            row.early_energy_slope,
        ),
        dtype=np.float64,
    )


def row_weights(rows: tuple[CalibrationRow, ...]) -> np.ndarray:
    counts = {
        family: sum(row.family == family for row in rows)
        for family in FAMILY_SHARES
        if any(row.family == family for row in rows)
    }
    denominator = sum(FAMILY_SHARES[family] for family in counts)
    return np.asarray(
        [FAMILY_SHARES[row.family] / counts[row.family] / denominator for row in rows],
        dtype=np.float64,
    )


def refit(rows: tuple[CalibrationRow, ...] = ROWS):
    x = np.asarray([features(row) for row in rows])
    y = np.asarray([endpoint_velocity(row) for row in rows])
    weights = row_weights(rows)
    means = np.sum(weights[:, None] * x, axis=0)
    scales = np.sqrt(np.sum(weights[:, None] * (x - means) ** 2, axis=0))
    standardized = (x - means) / scales
    coefficients = np.linalg.solve(
        standardized.T @ (weights[:, None] * standardized)
        + RIDGE_ALPHA * np.eye(standardized.shape[1]),
        standardized.T @ (weights * y),
    )
    intercept = np.sum(weights * y)
    return means, scales, intercept, coefficients, y, weights


def main() -> None:
    means, scales, intercept, coefficients, targets, weights = refit()
    x = np.asarray([features(row) for row in ROWS])
    predictions = intercept + ((x - means) / scales) @ coefficients
    endpoints = np.asarray([predict(row, velocity) for row, velocity in zip(ROWS, predictions)])
    actual_endpoints = np.asarray([row.final_rms for row in ROWS])
    print("feature means:", tuple(float(value) for value in means))
    print("feature scales:", tuple(float(value) for value in scales))
    print("intercept:", float(intercept))
    print("coefficients:", tuple(float(value) for value in coefficients))
    print("stage factors:", STAGE_FACTORS)
    print("energy retentions:", SQUEEZE_RETENTION)
    print(
        "weighted target MAPE:",
        float(np.sum(weights * np.abs(predictions - targets) / targets) * 100.0),
    )
    print(
        "weighted endpoint MAPE:",
        float(
            np.sum(weights * np.abs(endpoints - actual_endpoints) / actual_endpoints)
            * 100.0
        ),
    )
    target_lofo = 0.0
    endpoint_lofo = 0.0
    share_total = sum(FAMILY_SHARES.values())
    for family, family_share in FAMILY_SHARES.items():
        training_rows = tuple(row for row in ROWS if row.family != family)
        held_out_rows = tuple(row for row in ROWS if row.family == family)
        fold_means, fold_scales, fold_intercept, fold_coefficients, _, _ = refit(
            training_rows
        )
        fold_x = np.asarray([features(row) for row in held_out_rows])
        fold_targets = np.asarray([endpoint_velocity(row) for row in held_out_rows])
        fold_predictions = (
            fold_intercept
            + ((fold_x - fold_means) / fold_scales) @ fold_coefficients
        )
        fold_endpoints = np.asarray(
            [
                predict(row, velocity)
                for row, velocity in zip(held_out_rows, fold_predictions)
            ]
        )
        fold_actual_endpoints = np.asarray([row.final_rms for row in held_out_rows])
        target_lofo += family_share * float(
            np.mean(np.abs(fold_predictions - fold_targets) / fold_targets)
        )
        endpoint_lofo += family_share * float(
            np.mean(
                np.abs(fold_endpoints - fold_actual_endpoints)
                / fold_actual_endpoints
            )
        )
    print("leave-one-family-out target MAPE:", target_lofo / share_total * 100.0)
    print(
        "leave-one-family-out endpoint MAPE:",
        endpoint_lofo / share_total * 100.0,
    )


if __name__ == "__main__":
    main()
