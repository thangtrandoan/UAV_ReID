"""GASNet Stage 1 Evaluation Suite."""

from .datasets import load_vehicleid, load_veri776, load_vru_split, resolve_dataset_path
from .efficiency import (
    benchmark_latency_and_fps,
    count_parameters,
    measure_flops_and_macs,
    measure_peak_gpu_memory,
    run_full_efficiency_profile,
)
from .gate_benchmark import run_table_5_3_benchmark
from .metrics import compute_distance_matrix, evaluate_rank_map_minp, format_metrics_dict
from .models import extract_dataset_features, load_gasnet_variant
from .visualize import run_qualitative_evaluation

__all__ = [
    "resolve_dataset_path",
    "load_veri776",
    "load_vehicleid",
    "load_vru_split",
    "load_gasnet_variant",
    "extract_dataset_features",
    "evaluate_rank_map_minp",
    "compute_distance_matrix",
    "format_metrics_dict",
    "count_parameters",
    "measure_flops_and_macs",
    "benchmark_latency_and_fps",
    "measure_peak_gpu_memory",
    "run_full_efficiency_profile",
    "run_table_5_3_benchmark",
    "run_qualitative_evaluation",
]
