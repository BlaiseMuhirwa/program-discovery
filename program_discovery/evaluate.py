import argparse
from abstractions import (
    Variable,
    Expression,
    Statement,
    PruningAlgorithm,
    build_baseline_pruning_algorithm,
    ForEachStatement,
    IfElseStatement,
    AssignmentStatement,
    Node,
    CallableExpression,
    ALL_CALLABLES,
)
from plotting.metrics import compute_metrics
from copy import deepcopy
import flatnav
from flatnav.data_type import DataType
import numpy as np


class PruningProgramSearchExecutor:
    def __init__(
        self, pruning_algorithm: PruningAlgorithm, index: "flatnav.IndexFloatL2"
    ) -> None:
        self.algorithm = pruning_algorithm
        self.index = index
        self.distance_fn = self._create_distance_function()

    def _create_distance_function(self):
        def distance_fn(node1: Node, node2: Node) -> float:
            id1 = node1.id
            id2 = node2.id

            # Access through the C++ bindings
            return self.index.get_distance_between_nodes(id1, id2)

        return distance_fn

    def __call__(
        self, candidates: list[tuple[float, int]], M: int
    ) -> list[tuple[float, int]]:
        """
        Execute the pruning algorithm.
        NOTE: The list of candidates is retrieved from the C++ backend.
        :param candidates: List of (distance, node_id) tuples
        :param M: Maximum number of neighbors to select
        :returns: List of (distance, node_id) tuples (pruned candidates)
        """
        nodes = [Node(node_id=node_id, distance=dist) for dist, node_id in candidates]

        # set up execution environment
        env = {
            "input_candidates": nodes,
            # Implementation of take_first_M
            "take_first_M": lambda x: x[:M],
            "distance_fn": self.distance_fn,
        }

        # Add all the callable functions
        for fn in ALL_CALLABLES:
            if fn.__name__ not in env:
                env[fn.__name__] = fn

        # Execute the pruning algorithm
        for statement in self.algorithm._statements:
            self._execute_statement(statement, env)

        # Get the result
        result = env[self.algorithm._output_variable.name]

        # Convert back to format expected by the C++ backend
        return [(node.distance, node.node_id) for node in result[:M]]

    def _evaluate_expression(self, expression: Expression, env: dict) -> float:
        if isinstance(expression, Variable):
            return env[expression.name]

        elif isinstance(expression, CallableExpression):
            args = [env[arg.name] for arg in expression._input_args]
            fn = env[expression._transform_fn.__name__]
            return fn(*args)

    def _execute_statement(self, statement: Statement, env: dict) -> None:
        """
        Execute a statement in the pruning algorithm.
        """
        if isinstance(statement, AssignmentStatement):
            if isinstance(statement._right_side, Variable):
                env[statement._left_side.name] = env[statement._right_side.name]
            else:
                result = self._evaluate_expression(statement._right_side, env)
                env[statement._left_side.name] = result

        elif isinstance(statement, ForEachStatement):
            iterable = env[statement._iterable.name]
            for item in iterable:
                # print(f"Item: {item}")
                env[statement._item_variable.name] = item
                # Recursively execute each sub-statement
                for sub_stmt in statement._statements:
                    self._execute_statement(sub_stmt, env)

        elif isinstance(statement, IfElseStatement):
            condition = self._evaluate_expression(statement._condition, env)
            if condition:
                for sub_stmt in statement._if_statements:
                    self._execute_statement(sub_stmt, env)
            else:
                for sub_stmt in statement._else_statements:
                    self._execute_statement(sub_stmt, env)


def setup_index_with_pruning_algorithm(index, pruning_algorithm):
    executor = PruningProgramSearchExecutor(
        pruning_algorithm=pruning_algorithm, index=index
    )
    index.set_pruning_callback(callback=executor)
    return index


def evaluate_index(index, queries, ground_truth) -> dict:
    metrics = compute_metrics(
        requested_metrics=[
            "latency_p50",
            "latency_p95",
            "distance_computations",
            "recall",
            "qps",
        ],
        index=index,
        queries=queries,
        ground_truth=ground_truth,
        ef_search=100,
    )

    score = metrics["recall"] * metrics["qps"]
    metrics["score"] = score

    return metrics


def run_program_search(
    dataset: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    distance_type: str,
    index_data_type: DataType,
    dim: int,
    dataset_size: int,
    max_edges_per_node: int,
    num_iterations: int = 1000,
) -> tuple[str, dict]:
    """
    Run program search to discover the optimal pruning algorithm.
    """
    index = flatnav.index.create(
        distance_type=distance_type,
        index_data_type=index_data_type,
        dim=dim,
        dataset_size=dataset_size,
        max_edges_per_node=max_edges_per_node,
        verbose=True,
        collect_stats=False,
    )
    index.set_num_threads(4)
    # Create a pruning algorithm
    best_algorithm = build_baseline_pruning_algorithm()
    index = setup_index_with_pruning_algorithm(index, best_algorithm)

    index.add(dataset, ef_construction=100, num_initializations=100)
    baseline_metrics = evaluate_index(index, queries, ground_truth)

    best_metrics = baseline_metrics

    print(f"Baseline metrics: {baseline_metrics}")

    # Track all programs we've tried to avoid duplicates
    program_hashes = set()
    program_hashes.add(best_algorithm.to_str())

    for i in range(num_iterations):
        # Create a mutated copy of the algorithm
        candidate_algorithm = deepcopy(best_algorithm)

        attempts = 0
        while attempts < 10:
            candidate_algorithm.mutate(callables_library=ALL_CALLABLES)

            program_str = candidate_algorithm.to_str()
            if program_str not in program_hashes:
                program_hashes.add(program_str)
                break
            attempts += 1

        if attempts >= 10:
            print(f"Iteration {i+1}: Failed to find a unique mutation, skipping.")
            continue

        # Set up the index with the new algorithm
        del index
        index = flatnav.index.create(
            distance_type=distance_type,
            index_data_type=index_data_type,
            dim=dim,
            dataset_size=dataset_size,
            max_edges_per_node=max_edges_per_node,
            verbose=True,
            collect_stats=False,
        )
        index.set_num_threads(4)
        index = setup_index_with_pruning_algorithm(index, candidate_algorithm)
        index.add(dataset, ef_construction=100, num_initializations=100)

        print(f"Current algorithm: \n{candidate_algorithm.to_str()}")
        metrics = evaluate_index(index, queries, ground_truth)
        print(f"Current metrics: {metrics}")

        # Update if better
        if metrics["score"] > best_metrics["score"]:
            best_metrics = metrics
            best_algorithm = candidate_algorithm

            print(f"Iteration {i+1}: {metrics}")
            print(f"New best algorithm: \n{best_algorithm.to_str()}")

    return best_algorithm, best_metrics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Flatnav on Big ANN datasets."
    )

    parser.add_argument(
        "--num-node-links",
        nargs="+",
        type=int,
        default=[16, 32],
        help="Number of node links per node.",
    )

    parser.add_argument(
        "--ef-construction",
        nargs="+",
        type=int,
        default=[100, 200, 300, 400, 500],
        help="ef_construction parameter.",
    )

    parser.add_argument(
        "--ef-search",
        nargs="+",
        type=int,
        default=[100, 200, 300, 400, 500, 1000, 2000, 3000, 4000],
        help="ef_search parameter.",
    )

    parser.add_argument(
        "--num-initializations",
        required=False,
        nargs="+",
        type=int,
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a single ANNS benchmark dataset to run on.",
    )
    parser.add_argument(
        "--queries", required=True, help="Path to a singe queries file."
    )
    parser.add_argument(
        "--gtruth",
        required=True,
        help="Path to a single ground truth file to evaluate on.",
    )
    parser.add_argument(
        "--metric",
        required=True,
        default="l2",
        help="Distance tye. Options include `l2` and `angular`.",
    )

    parser.add_argument(
        "--num-build-threads",
        required=False,
        default=1,
        type=int,
        help="Number of threads to use during index construction.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    dataset = np.load(args.dataset)
    queries = np.load(args.queries)
    ground_truth = np.load(args.gtruth)

    best_algorithm, best_metrics = run_program_search(
        dataset=dataset,
        queries=queries,
        ground_truth=ground_truth,
        distance_type=args.metric,
        index_data_type=DataType.float32,
        dim=dataset.shape[1],
        dataset_size=dataset.shape[0],
        max_edges_per_node=16,
        num_iterations=1000,
    )

    print(f"Best algorithm: {best_algorithm.to_str()}")
    print(f"Best metrics: {best_metrics}")


if __name__ == "__main__":
    main()
