import numpy as np
import matplotlib.pyplot as plt


def custom_diff(hist, ignore_range):
    valid_indices = np.where(~ignore_range)[0]
    effective_diff = np.zeros_like(hist, dtype=float)
    for i in range(len(valid_indices) - 1):
        curr_idx = valid_indices[i]
        next_idx = valid_indices[i + 1]
        effective_diff[curr_idx] = hist[next_idx] - hist[curr_idx]
    return effective_diff


def detect_outliers_with_visualization(weights, bin_width=0.05, iterations=2, threshold=5):
    bins = np.arange(min(weights), max(weights) + bin_width, bin_width)
    hist, bin_edges = np.histogram(weights, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total_points = len(weights)
    cumulative_hist = np.cumsum(hist) / total_points
    ignore_range_ori = (cumulative_hist >= 0.15) & (cumulative_hist <= 0.85)
    ignore_range = (cumulative_hist >= 0.15) & (cumulative_hist <= 0.85)
    all_detected_anomalies = np.zeros_like(hist, dtype=bool)

    for iteration in range(iterations):
        freq_diff = custom_diff(hist, ignore_range)
        detected_anomalies = np.zeros_like(hist, dtype=bool)
        valid_indices = np.where(~ignore_range)[0]
        for i in range(len(valid_indices) - 1):
            j = valid_indices[i]
            if bin_centers[j] < 0:
                if freq_diff[j] < -threshold:
                    detected_anomalies[j] = True
            elif bin_centers[j] > 0:
                if freq_diff[j] > threshold:
                    detected_anomalies[valid_indices[i + 1]] = True

        all_detected_anomalies |= detected_anomalies
        plt.figure(figsize=(10, 6))
        plt.hist(weights, bins=bins, alpha=0.75, color='blue', edgecolor='black', label='All weights')
        for i in range(len(ignore_range)):
            if ignore_range_ori[i]:
                plt.axvspan(bin_edges[i], bin_edges[i + 1], color='gray', alpha=0.3,
                            label='Ignored range' if i == 0 else None)
            if (ignore_range_ori ^ ignore_range)[i]:
                plt.axvspan(bin_edges[i], bin_edges[i + 1], color='red', alpha=0.3,
                            label='Anomaly range' if i == 0 else None)
        plt.scatter(bin_centers[detected_anomalies], hist[detected_anomalies], color='red',
                    label='Detected anomalies')
        plt.title(f'Iteration {iteration + 1}: Weight Distribution with Anomaly Detection')
        plt.xlabel('Weight value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.legend()
        plt.show()
        ignore_range |= detected_anomalies

    return all_detected_anomalies


def detect_outliers_with_mask(name, weights, bin_width=0.05, iterations=2, threshold=5, ratio=0.001):
    bins = np.arange(min(weights), max(weights) + bin_width, bin_width)
    hist, bin_edges = np.histogram(weights, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total_points = len(weights)
    cumulative_hist = np.cumsum(hist) / total_points
    ignore_range_ori = (cumulative_hist >= ratio) & (cumulative_hist <= 1 - ratio)
    ignore_range = (cumulative_hist >= ratio) & (cumulative_hist <= 1 - ratio)
    all_detected_anomalies = np.zeros_like(hist, dtype=bool)
    for iteration in range(iterations):
        freq_diff = custom_diff(hist, ignore_range)
        detected_anomalies = np.zeros_like(hist, dtype=bool)
        valid_indices = np.where(~ignore_range)[0]
        for i in range(1, len(valid_indices)):
            if valid_indices[i] - valid_indices[i - 1] > 1:
                valid_indices = np.delete(valid_indices, [i - 1, i])
                break
        for i in range(len(valid_indices) - 1):
            j = valid_indices[i]
            if bin_centers[j] < 0:
                if freq_diff[j] < -threshold:
                    detected_anomalies[j] = True
            elif bin_centers[j] > 0:
                if freq_diff[j] > threshold:
                    detected_anomalies[valid_indices[i + 1]] = True
        all_detected_anomalies |= detected_anomalies
        if iteration == iterations - 1:
            plt.figure(figsize=(10, 6))
            plt.hist(weights, bins=bins, alpha=0.75, color='blue', edgecolor='black', label='All weights')
            for i in range(len(ignore_range)):
                if ignore_range_ori[i]:
                    plt.axvspan(bin_edges[i], bin_edges[i + 1], color='gray', alpha=0.3,
                                label='Ignored range' if i == 0 else None)
                if (ignore_range_ori ^ ignore_range)[i]:
                    plt.axvspan(bin_edges[i], bin_edges[i + 1], color='red', alpha=0.3,
                                label='Anomaly range' if i == 0 else None)
            plt.scatter(bin_centers[detected_anomalies], hist[detected_anomalies], color='red',
                        label='Detected anomalies')
            plt.title(f'{name}: Weight Distribution with Anomaly Detection')
            plt.xlabel('Weight value')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.legend()
            # plt.show()
            plt.xlim(-0.8, 0.8)
            plt.ylim(0, 100)
            if iteration == iterations - 1:
                # plt.show()
                plt.savefig(f'imgs/{name}.png')
                plt.close()
        ignore_range |= detected_anomalies
    outlier_indices = []
    for i in range(len(bin_centers)):
        mask_condition = (weights >= bin_edges[i]) & (weights < bin_edges[i + 1])
        if all_detected_anomalies[i]:
            outlier_indices.extend(np.where(mask_condition)[0])
    return np.array(outlier_indices)


if __name__ == '__main__':
    # test
    weights = np.concatenate([
        np.random.normal(loc=0, scale=0.5, size=1000),  # Unimodal distribution
        [1.5] * 50,  # Outlier region
        [-1.2] * 30,
        [1.6] * 30,
        [-1.3] * 20,
    ])
    outlier_indices = detect_outliers_with_visualization(weights, bin_width=0.1, iterations=3)