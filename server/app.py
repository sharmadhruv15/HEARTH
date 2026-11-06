"""
Hearth — Flower Federated Server
Aggregates ONLY shared LoRA weights (attention layers) from each client.
Personal weights (MLP layers) never reach this server.
Sequential mode: one client per server instance, 3 rounds per client.
"""

import flwr as fl
import numpy as np
from typing import List, Tuple, Optional, Dict
from flwr.common import (
    Parameters, FitRes, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class PersonalizedFedAvg(FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        print(f"\n[Server] Round {server_round} — aggregating {len(results)} client(s)")

        all_weights = []
        total_examples = 0
        for _, fit_res in results:
            num_examples = fit_res.num_examples
            arrays = parameters_to_ndarrays(fit_res.parameters)
            all_weights.append((arrays, num_examples))
            total_examples += num_examples
            print(f"  [Server] Received {len(arrays)} shared weight tensors "
                  f"({num_examples} examples)")

        averaged = []
        for layer_idx in range(len(all_weights[0][0])):
            layer_avg = np.sum(
                [w[layer_idx] * (n / total_examples) for w, n in all_weights],
                axis=0
            )
            averaged.append(layer_avg)

        print(f"  [Server] Aggregation complete — {total_examples} total examples.")
        print(f"  [Server] Personal weights: NOT received, NOT aggregated. Privacy preserved.\n")

        return ndarrays_to_parameters(averaged), {"round": server_round}

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}


def main():
    strategy = PersonalizedFedAvg(
        min_fit_clients=1,
        min_evaluate_clients=0,
        min_available_clients=1,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
    )

    print("[Server] Starting Hearth Federated Server on port 8080")
    print("[Server] 3 rounds for this client.")
    print("[Server] Only shared LoRA weights will be aggregated.")
    print("[Server] Personal weights stay on each client — never transmitted.\n")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()