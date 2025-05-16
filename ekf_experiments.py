import subprocess
import numpy as np
import matplotlib.pyplot as plt

r_values = [1/64, 1/16, 1/4, 4, 16, 64]
n_trials = 10

def run_trial(r, mode):
    errors = []
    mahalanobis_errors = []
    anees = []

    for seed in range(n_trials):
        if mode == 'a':
            args = [
                'python', 'localization.py', 'ekf',
                '--data-factor', str(r),
                '--filter-factor', str(r),
                '--seed', str(seed)
            ]
        elif mode == 'b':
            args = [
                'python', 'localization.py', 'ekf',
                '--filter-factor', str(r),
                '--seed', str(seed)
            ]
        else:
            raise ValueError("Mode must be 'a' or 'b'")

        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        for line in result.stdout.split('\n'):
            if 'Mean position error' in line:
                errors.append(float(line.split(':')[1]))
            elif 'Mean Mahalanobis error' in line:
                mahalanobis_errors.append(float(line.split(':')[1]))
            elif 'ANEES' in line:
                anees.append(float(line.split(':')[1]))

    return np.mean(errors), np.mean(mahalanobis_errors), np.mean(anees)


def run_experiments(mode='a'):
    mean_errors = []
    mean_mahal_errors = []
    mean_anees = []

    for r in r_values:
        print(f"Running for r = {r} (mode {mode})")
        error, mahal_error, anees_val = run_trial(r, mode)
        mean_errors.append(error)
        mean_mahal_errors.append(mahal_error)
        mean_anees.append(anees_val)

    # Plot 1: Mean Position Error
    plt.figure(figsize=(8, 4))
    plt.plot(r_values, mean_errors, marker='o', color='blue')
    plt.xscale('log')
    plt.xlabel('r (log scale)')
    plt.ylabel('Mean Position Error')
    plt.title(f'Part {mode.upper()} - Mean Position Error vs Noise Factor')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'ekf_part_{mode}_mean_position_error.png')
    plt.show()

    # Plot 2: Mean Mahalanobis Error
    plt.figure(figsize=(8, 4))
    plt.plot(r_values, mean_mahal_errors, marker='^', color='green')
    plt.xscale('log')
    plt.xlabel('r (log scale)')
    plt.ylabel('Mean Mahalanobis Error')
    plt.title(f'Part {mode.upper()} - Mean Mahalanobis Error vs Noise Factor')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'ekf_part_{mode}_mean_mahalanobis_error.png')
    plt.show()

    # Plot 3: ANEES
    plt.figure(figsize=(8, 4))
    plt.plot(r_values, mean_anees, marker='x', color='red')
    plt.xscale('log')
    plt.xlabel('r (log scale)')
    plt.ylabel('ANEES')
    plt.title(f'Part {mode.upper()} - ANEES vs Noise Factor')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'ekf_part_{mode}_anees.png')
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['a', 'b'], required=True, help="Choose part 'a' or 'b'")
    args = parser.parse_args()

    run_experiments(mode=args.mode)
    # Save the computed data for later comparison
    np.save(f'ekf_part_{mode}_pos_error.npy', mean_errors)
    np.save(f'ekf_part_{mode}_mahal_error.npy', mean_mahal_errors)
    np.save(f'ekf_part_{mode}_anees.npy', mean_anees)
