import joblib
import json
import os
import numpy as np


def export_forest_to_cpp(model_path="ml/models/rf_scheduler.pkl",
                         output_path="ml/models/gbm_model.json"):
    clf = joblib.load(model_path)
    forest_data = {"n_estimators": len(clf.estimators_), "trees": []}

    for tree_idx, estimator in enumerate(clf.estimators_):
        tree = estimator.tree_
        tree_data = {
            "n_nodes": tree.node_count,
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "children_left": tree.children_left.tolist(),
            "children_right": tree.children_right.tolist(),
            "value": []
        }
        for i in range(tree.node_count):
            class_counts = tree.value[i][0].tolist()
            tree_data["value"].append(class_counts)

        forest_data["trees"].append(tree_data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(forest_data, f, indent=2)

    print(f"Exported {len(clf.estimators_)} trees to {output_path}")
    print(f"  Total nodes: {sum(t['n_nodes'] for t in forest_data['trees'])}")


def generate_cpp_header(model_path="ml/models/gbm_model.json",
                        header_path="ml/models/compiled_forest.h"):
    with open(model_path) as f:
        forest = json.load(f)

    lines = [
        "#ifndef COMPILED_FOREST_H",
        "#define COMPILED_FOREST_H",
        "",
        "#include <vector>",
        "#include <algorithm>",
        "",
        "namespace compiled_forest {",
        "",
        f"static const int N_TREES = {forest['n_estimators']};",
        "",
    ]

    for ti, tree in enumerate(forest["trees"]):
        n = tree["n_nodes"]
        feat = tree["feature"]
        thresh = tree["threshold"]
        left = tree["children_left"]
        right = tree["children_right"]
        vals = tree["value"]

        lines.append(f"inline int predict_tree_{ti}(const double* features) {{")
        lines.append(f"    int node = 0;")
        lines.append(f"    while (true) {{")

        lines.append(f"        switch (node) {{")
        for i in range(n):
            lines.append(f"            case {i}:")
            if left[i] == -1:
                class_0 = vals[i][0]
                class_1 = vals[i][1] if len(vals[i]) > 1 else 0
                pred = 0 if class_0 >= class_1 else 1
                lines.append(f"                return {pred};")
            else:
                lines.append(f"                node = (features[{feat[i]}] <= {thresh[i]}) ? {left[i]} : {right[i]};")
                lines.append(f"                break;")
        lines.append(f"            default: return 0;")
        lines.append(f"        }}")
        lines.append(f"    }}")
        lines.append(f"}}")
        lines.append("")

    lines.append("inline int predict(const double* features) {")
    lines.append("    int votes_gpu = 0;")
    for ti in range(forest["n_estimators"]):
        lines.append(f"    votes_gpu += predict_tree_{ti}(features);")
    lines.append(f"    return (votes_gpu > N_TREES / 2) ? 1 : 0;")
    lines.append("}")
    lines.append("")
    lines.append("}")
    lines.append("")
    lines.append("#endif")

    with open(header_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated C++ header at {header_path}")


if __name__ == "__main__":
    export_forest_to_cpp()
    generate_cpp_header()
