"""
key_rate.py

Computes a GLLP-style secure key rate "cost label" for each (link, epoch) row
of telemetry, to be used as ground truth for training/evaluating the GCN.

This is the simplified single-intensity BB84 case (Qmu ~= Q1, Emu ~= e1),
consistent with the current simulator (starter.py), which does not yet
implement multi-intensity decoy-state transmission. If/when starter.py is
updated to sweep multiple (mu, nu) intensities, replace estimate_q1_e1()
with a real decoy-state estimator before computing the key rate.

R_GLLP = q * { -f(E) * Q * h2(E) + Q1 * [1 - h2(e1)] }

Where, under the single-intensity simplification used here:
    Q  ~= sifted gain, approximated from (signal_count + decoy_count)
    E  ~= QBER (observed)
    Q1 ~= Q  (single-photon gain approximated by total observed gain)
    e1 ~= E  (single-photon QBER approximated by total observed QBER)
    f(E) = 1.16 (typical Cascade/LDPC error-correction inefficiency)
    q    = 0.5 (standard BB84 basis-sifting factor)
"""

import math
import csv

Q_SIFT = 0.5          # basis-sifting factor for standard BB84
F_EC = 1.16           # error-correction inefficiency (Cascade-like)
PULSE_NORM = 10000.0  # normalization constant matching starter.py's base signal_count


def h2(x: float) -> float:
    """Binary Shannon entropy. Defined as 0 at x=0 or x=1."""
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def estimate_q1_e1(qber: float, signal_count: int, decoy_count: int):
    """
    Single-intensity simplification: treat the observed gain/QBER as direct
    estimates of the single-photon gain/QBER (Q1, e1). Replace this with a
    true decoy-state estimator if/when multi-intensity data is available.
    """
    q_mu = (signal_count + decoy_count) / PULSE_NORM
    e_mu = qber
    q1 = q_mu
    e1 = e_mu
    return q_mu, e_mu, q1, e1


def secure_key_rate(qber: float, signal_count: int, decoy_count: int) -> float:
    """Returns the GLLP secure key rate (bits per pulse-normalized unit).
    Returns 0.0 (key rate floor) if the bound goes negative, since a
    negative rate has no physical meaning -- it indicates no secure key
    can be extracted under current channel conditions."""
    q_mu, e_mu, q1, e1 = estimate_q1_e1(qber, signal_count, decoy_count)
    rate = Q_SIFT * (-F_EC * q_mu * h2(e_mu) + q1 * (1 - h2(e1)))
    return max(rate, 0.0)


def key_loss_cost(qber: float, signal_count: int, decoy_count: int, is_attacked: int) -> float:
    """
    Ground-truth cost label: secure-key-rate *loss* relative to a clean-channel
    baseline (QBER ~ 0.02, the natural-noise floor used in starter.py).
    This is the quantity the GCN's custom loss function should penalize
    against, since it captures real key-rate impact rather than a raw
    classification error.
    """
    baseline_rate = secure_key_rate(0.02, signal_count, decoy_count)
    actual_rate = secure_key_rate(qber, signal_count, decoy_count)
    loss = max(baseline_rate - actual_rate, 0.0)
    return loss


def process_file(input_csv: str, output_csv: str):
    with open(input_csv, newline="") as f_in, open(output_csv, "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames + ["Secure_Key_Rate", "Key_Rate_Cost"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            qber = float(row["QBER"])
            signal_count = int(row["Signal_Count"])
            decoy_count = int(row["Decoy_Count"])
            is_attacked = int(row["Attacked_Flag"])

            row["Secure_Key_Rate"] = round(secure_key_rate(qber, signal_count, decoy_count), 6)
            row["Key_Rate_Cost"] = round(key_loss_cost(qber, signal_count, decoy_count, is_attacked), 6)
            writer.writerow(row)

    print(f"Wrote ground-truth cost labels to {output_csv}")


if __name__ == "__main__":
    process_file(
        input_csv="sequence_telemetry_output.csv",
        output_csv="sequence_telemetry_with_costs.csv",
    )
