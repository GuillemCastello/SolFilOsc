from .decorators import benchmark
from .filename_helpers import time_offsets_seconds


@benchmark
def create_tdeltas_array(file_list):
    tdeltas = time_offsets_seconds(file_list)
    print("tdeltas array calculated")
    return tdeltas
