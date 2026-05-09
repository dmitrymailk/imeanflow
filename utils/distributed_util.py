import os

import jax


def initialize_distributed():
    """Initialize JAX distributed mode for both single-host and explicit multi-host runs."""
    coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS")
    num_processes = int(os.environ.get("JAX_NUM_PROCESSES", "1"))
    process_id = int(os.environ.get("JAX_PROCESS_ID", "0"))

    if coordinator_address is None:
        if num_processes != 1 or process_id != 0:
            raise ValueError(
                "JAX_COORDINATOR_ADDRESS must be set when using multi-process distributed runs."
            )
        coordinator_address = "127.0.0.1:12355"

    try:
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id,
        )
    except RuntimeError as exc:
        if "already been initialized" not in str(exc):
            raise
