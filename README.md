# EKF and PF for Localization of a Mobile Robot

The objective of the practical work is to understand how the Extended Kalman Filter (EKF) and the Particle Filter (PF) work for localizing a mobile robot and to develop an implementation of each. 



## How to Run the Experiments

This project contains scripts to run EKF and PF localization experiments based on different noise scaling factors. Below are instructions for running each set of experiments.

### Prerequisites

Ensure you have Python 3 installed, along with the required libraries:

```bash
pip install numpy matplotlib
```

Make sure `localization.py` exists in the same directory and is properly implemented to accept the expected arguments.

---

### EKF Experiments

Run experiments for part **b** or **c** using:

```bash
python ekf_experiments.py --mode b
```

or

```bash
python ekf_experiments.py --mode c
```

This will:

* Run the EKF localization multiple times for each `r` value.
* Generate and save plots for:

  * Mean Position Error
  * Mean Mahalanobis Error
  * ANEES

---

### PF Experiments

Run experiments for part **b**, **c**, or **d** using:

```bash
python pf_experiments.py --mode b
```

```bash
python pf_experiments.py --mode c
```

```bash
python pf_experiments.py --mode d
```

* **Mode b** and **c**: Similar to EKF, these will test the Particle Filter with different noise scaling factors.
* **Mode d**: Tests with different particle counts (`20`, `50`, `500`).

Each run will:

* Execute `localization.py` using subprocess.
* Run multiple trials (default: 10).
* Generate and save plots for:

  * Mean Position Error
  * Mean Mahalanobis Error
  * ANEES

---

### Output

All plots will be saved as PNG files in the working directory:

* `ekf_part_b_mean_position_error.png`, etc.
* `pf_part_d_anees.png`, etc.
