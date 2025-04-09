from phylo_gnn.data.feature_extraction import (
    FeaturePipeline,
    NodeNames,
    EdgeNames,
)


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
        NodeNames.NODE.value: [
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
        (
            NodeNames.NODE.value,
            EdgeNames.HAS_PARENT.value,
            NodeNames.NODE.value,
        ): [FeaturePipeline("distances", "log1p")],
        (
            NodeNames.NODE.value,
            EdgeNames.HAS_CHILD.value,
            NodeNames.NODE.value,
        ): [FeaturePipeline("distances", "log1p")],
    }


def node_features_with_level_nodes() -> dict[str, list[FeaturePipeline]]:
    node_features = default_node_features()
    node_features[NodeNames.LEVEL.value] = [
        FeaturePipeline("num_nodes_by_level", "div_by_avg_num_nodes_by_level"),
        FeaturePipeline("num_nodes_by_level", "log1p"),
        FeaturePipeline("level_node_id", "div_by_max_level"),
        FeaturePipeline("level_node_id", "log1p"),
        # ---- AVG DISTANCE FEATURES----
        FeaturePipeline("avg_distance_to_root_by_level", "log1p"),
        FeaturePipeline("avg_distance_to_leaves_by_level", "log1p"),
        FeaturePipeline(
            "avg_topological_distance_to_leaves_by_level",
            "div_by_max_level",
        ),
        FeaturePipeline(
            "avg_topological_distance_to_root_by_level",
            "div_by_max_level",
        ),
        FeaturePipeline("avg_branch_lengths_by_level", "log1p"),
        # ---- MAX DISTANCE FEATURES----
        FeaturePipeline("max_distance_to_root_by_level", "log1p"),
        FeaturePipeline("max_distance_to_leaves_by_level", "log1p"),
        FeaturePipeline(
            "max_topological_distance_to_leaves_by_level",
            "div_by_max_level",
        ),
        FeaturePipeline(
            "max_topological_distance_to_root_by_level",
            "div_by_max_level",
        ),
        FeaturePipeline("max_branch_lengths_by_level", "log1p"),
        # ---- MIN DISTANCE FEATURES----
        FeaturePipeline("min_distance_to_root_by_level", "log1p"),
        FeaturePipeline("min_distance_to_leaves_by_level", "log1p"),
        FeaturePipeline(
            "min_topological_distance_to_leaves_by_level",
            "div_by_max_level",
        ),
        FeaturePipeline(
            "min_topological_distance_to_root_by_level",
            "div_by_max_level",
        ),
        FeaturePipeline("min_branch_lengths_by_level", "log1p"),
        # ---- STD DISTANCE FEATURES----
        FeaturePipeline("std_distance_to_root_by_level", None),
        FeaturePipeline("std_distance_to_leaves_by_level", None),
        FeaturePipeline(
            "std_topological_distance_to_leaves_by_level",
            None,
        ),
        FeaturePipeline(
            "std_topological_distance_to_root_by_level",
            None,
        ),
        FeaturePipeline("std_branch_lengths_by_level", None),
    ]
    return node_features


def edge_attributes_with_level_nodes(
    only_node2level: bool = False,
) -> dict[tuple[str, str, str], list[FeaturePipeline]]:
    edge_attributes = default_edge_attributes()
    if not only_node2level:
        edge_attributes[
            (
                NodeNames.LEVEL.value,
                EdgeNames.HAS_NODE.value,
                NodeNames.NODE.value,
            )
        ] = [
            FeaturePipeline(
                "level2node_position_in_level", "div_by_num_nodes_in_level"
            )
        ]
    edge_attributes[
        (
            NodeNames.NODE.value,
            EdgeNames.IS_IN_LEVEL.value,
            NodeNames.LEVEL.value,
        )
    ] = [
        FeaturePipeline(
            "node2level_position_in_level", "div_by_num_nodes_in_level"
        )
    ]
    return edge_attributes


if __name__ == "__main__":
    # Print len for each node type of node_features_with_level_nodes
    _node_features = node_features_with_level_nodes()
    for node_type, _features in _node_features.items():
        print(f"{node_type}: {len(_features)}")
