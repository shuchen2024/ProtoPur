import numpy as np


def assign_clusters_by_distance_and_coverage_v3(
    dist,                     # (k, C) 每个 cluster 到 C 个类顶点的距离，列顺序与 class_order 一致
    bag_labels,               # (B,) 每个 bag 的标签，取值在 class_order 中
    bag_cluster_counts,       # (B, k) bag i 在 cluster j 中的 patch 数
    patch_threshold=1,
    class_order=(1, 2, 3),
    *,
    use_nearest_gate=True,
    dist_max=None,            # 到该类顶点的距离上限；超过则不再为该类分配（用于剪掉离顶点过远的杂 cluster）
    return_info=False,
):
    """改进版 assign_clusters_by_distance_and_coverage_new2。

    与原版的差异（仅 bug fix + 可选剪枝，整体语义不变）：

      1) [BUG FIX] 当 cluster 的 nearest != 当前类时，原版用 `break`，但排序是按
         dist[:, c_idx] 升序、与 nearest 单调性无关。break 会漏掉后续 nearest 仍指
         向当前类的 cluster。这里改为 `continue`，只跳过该 cluster，继续检查更远的。

      2) [新增可选] dist_max：一个 cluster 即使 nearest=c，但若其到 c 顶点的距离过
         大（杂质多），也不再为它打 c 类伪标签。由于排序按距离升序，超限即可 break。

      3) [新增可选] return_info：返回每类分配的 cluster 数 / 未覆盖 bag，便于调参。

    保留的核心语义：
      - 按 dist[:, c_idx] 升序依次为每个类挑 cluster
      - 覆盖到该类所有 bag（每个 bag 累计 patch 数 >= patch_threshold）即停
      - 未被任何类挑走的 cluster 留 label 0（即"negative / 背景"伪标签的来源）
    """
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