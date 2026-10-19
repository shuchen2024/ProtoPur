import numpy as np


def assign_clusters_by_distance_and_coverage_v3(
    dist,                
    bag_labels,             
    bag_cluster_counts,     
    patch_threshold=1,
    class_order=(1, 2, 3),
    *,
    use_nearest_gate=True,
    dist_max=None,          
    return_info=False,
):
    dist = np.asarray(dist)
    bag_labels = np.asarray(bag_labels)
    bag_cluster_counts = np.asarray(bag_cluster_counts)

    k, num_vertices = dist.shape
    assert num_vertices == len(class_order), "dist 第二维应等于 class_order 长度"

    cluster_labels = np.zeros(k, dtype=int)
    nearest_class_idx = dist.argmin(axis=1)
    class_to_vertex_index = {cls: i for i, cls in enumerate(class_order)}

    info = {
        'uncovered_bags_per_class': {},
        'num_clusters_per_class': {},
    }

    for cls in class_order:
        c_idx = class_to_vertex_index[cls]
        bags_of_cls = np.where(bag_labels == cls)[0]

        if bags_of_cls.size == 0:
            info['uncovered_bags_per_class'][cls] = np.array([], dtype=int)
            info['num_clusters_per_class'][cls] = 0
            continue

        remaining_clusters = np.where(cluster_labels == 0)[0]
        if remaining_clusters.size == 0:
            info['uncovered_bags_per_class'][cls] = bags_of_cls
            info['num_clusters_per_class'][cls] = 0
            continue

        sort_idx = np.argsort(dist[remaining_clusters, c_idx], kind='stable')
        remaining_sorted = remaining_clusters[sort_idx]

        cumulative_counts = np.zeros(bags_of_cls.size, dtype=np.int64)
        covered = np.zeros(bags_of_cls.size, dtype=bool)
        n_assigned = 0

        for cl in remaining_sorted:
            d_c = dist[cl, c_idx]

            if dist_max is not None and d_c > dist_max:
                break

            if use_nearest_gate and nearest_class_idx[cl] != c_idx:
                continue

            cluster_labels[cl] = cls
            counts_for_these_bags = bag_cluster_counts[bags_of_cls, cl]
            cumulative_counts += counts_for_these_bags
            covered = cumulative_counts >= patch_threshold
            n_assigned += 1

            if covered.all():
                break

        info['uncovered_bags_per_class'][cls] = bags_of_cls[~covered]
        info['num_clusters_per_class'][cls] = n_assigned

    if return_info:
        return cluster_labels, info
    return cluster_labels
