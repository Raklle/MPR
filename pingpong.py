#!/usr/bin/env python3
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 2:
        print("Expected only two nodes")


N = 100000
sizes = [1, 8, 16]


def pingpong_ready(msg_size, iterations):
    buf = bytearray(msg_size)

    comm.Barrier()
    start = MPI.Wtime()

    if rank == 0:
        for _ in range(iterations):
            req = comm.irecv(source=1)
            comm.Rsend(buf, dest=1)
            req.wait()
    else:
        for _ in range(iterations):
            req = comm.irecv(source=0)
            req.wait()
            comm.Rsend(buf, dest=0)

    end = MPI.Wtime()
    return (end - start) / (2 * iterations)


for s in sizes:
    t = pingpong_ready(s, N)
    if rank == 0:
        print(f"size:{s:9d}  time:{t:.9e}")

lat = pingpong_ready(sizes[0], N)

if rank == 0:
    print("\nopoznienie:", f"{lat:.9e}", "s")