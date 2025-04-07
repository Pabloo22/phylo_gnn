from phylo_gnn.data.feature_extraction import FeaturePipeline


def default_node_features() -> dict[str, list[FeaturePipeline]]:
    """
    - node:
        - feature_name: "branch_lengths"
            normalization_fn_name: "log1p"
        - feature_name: "distance_to_root"
            normalization_fn_name: "log1p"
        - feature_name: "distance_to_leaves"
            normalization_fn_name: "log1p"
        - feature_name: "topological_distance_to_leaves"
            normalization_fn_name: "div_by_max_level"
        - feature_name: "topological_distance_to_root"
            normalization_fn_name: "div_by_max_level"
        - feature_name: "levels"
            normalization_fn_name: "div_by_max_level"
        - feature_name: "position_in_level"
            normalization_fn_name: "div_by_num_nodes_in_level"
    """
    return {
        "node": [
            FeaturePipeline("branch_lengths", "log1p"),
            FeaturePipeline("distance_to_root", "log1p"),
            FeaturePipeline("distance_to_leaves", "log1p"),
            FeaturePipeline(
                "topological_distance_to_leaves", "div_by_max_level"
            ),
            FeaturePipeline(
                "topological_distance_to_root", "div_by_max_level"
            ),
            FeaturePipeline("levels", "div_by_max_level"),
            FeaturePipeline("position_in_level", "div_by_num_nodes_in_level"),
        ]
    }


def default_edge_attributes() -> (
    dict[tuple[str, str, str], list[FeaturePipeline]]
):
    return {
        ("node", "has_parent", "node"): [
            FeaturePipeline("distances", "log1p")
        ],
        ("node", "has_child", "node"): [FeaturePipeline("distances", "log1p")],
    }
