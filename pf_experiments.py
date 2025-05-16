import subprocess
import numpy as np
import matplotlib.pyplot as plt

r_values = [1/64, 1/16, 1/4, 4, 16, 64]
n_trials = 10
particle_counts = [20, 50, 500]  # Only used in part d


def run_trial(r, mode, num_particles=None):
    errors, mahal_errors, anees_vals = [], [], []

    for seed in range(n_trials):
        args = ['python', 'localization.py', 'pf',
                '--seed', str(seed)]

        if mode == 'a':
            args += ['--data-factor', str(r), '--filter-factor', str(r)]
        elif mode == 'b':
            args += ['--filter-factor', str(r)]
        elif mode == 'c':
            args += ['--filter-factor', str(r)]
            if num_particles:
                args += ['--num-particles', str(num_particles)]
        else:
            raise ValueError("Mode must be 'a', 'b', or 'c'")

        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        for line in result.stdout.split('\n'):
            if 'Mean position error' in line:
                errors.append(float(line.split(':')[1]))
            elif 'Mean Mahalanobis error' in line:
                mahal_errors.append(float(line.split(':')[1]))
            elif 'ANEES' in line:
                anees_vals.append(float(line.split(':')[1]))

    return np.mean(errors), np.mean(mahal_errors), np.mean(anees_vals)


def plot_metrics(r_vals, metric_sets, labels, title, filename):
    plt.figure(figsize=(8, 6))
    for metrics, label in zip(metric_sets, labels):
        plt.plot(r_vals, metrics, marker='o', label=label)

    plt.xscale('log')
    plt.xlabel('r (log scale)')
    plt.ylabel('Metric Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def run_experiment(mode):
    if mode in ['a', 'b']:
        mean_errors, mean_mahal, mean_anees = [], [], []

        for r in r_values:
            print(f"Running for r = {r} (mode {mode})")
            err, mahal, anees = run_trial(r, mode)
            mean_errors.append(err)
            mean_mahal.append(mahal)
            mean_anees.append(anees)

        # Plot separately
        plot_metrics(r_values, [mean_errors], ['Mean Position Error'],
                    f'PF Part {mode.upper()} - Position Error', f'pf_part_{mode}_pos_error.png')
        plot_metrics(r_values, [mean_mahal], ['Mean Mahalanobis Error'],
                    f'PF Part {mode.upper()} - Mahalanobis Error', f'pf_part_{mode}_mahal_error.png')
        plot_metrics(r_values, [mean_anees], ['ANEES'],
                    f'PF Part {mode.upper()} - ANEES', f'pf_part_{mode}_anees.png')

    elif mode == 'c':
        all_errors = []
        all_anees = []

        for num_particles in particle_counts:
            errors = []
            anees = []
            print(f"\nParticles = {num_particles}")
            for r in r_values:
                print(f"  r = {r}")
                err, _, ane = run_trial(r, mode, num_particles)
                errors.append(err)
                anees.append(ane)
            all_errors.append(errors)
            all_anees.append(anees)

        labels = [f'N = {n}' for n in particle_counts]
        plot_metrics(r_values, all_errors, labels,
                    'PF Part D - Position Error vs r for Different Particle Counts',
                    'pf_part_d_pos_error.png')
        plot_metrics(r_values, all_anees, labels,
                    'PF Part D - ANEES vs r for Different Particle Counts',
                    'pf_part_d_anees.png')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['a', 'b', 'c'], required=True,
                        help="Choose experiment part: a, b, or c")
    args = parser.parse_args()
    run_experiment(args.mode)
