#!/usr/bin/env python3
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


N = 100000
sizes = [1, 8, 16, 1024]


def pingpong_ssend(msg_size, i):
    send_buf = bytearray(msg_size)
    recv_buf = bytearray(msg_size)

    start = MPI.Wtime()

    for _ in range(i):
        if rank == 0:
            comm.Ssend(send_buf, dest=1)
            comm.Recv(recv_buf, source=1)
        else:
            comm.Recv(recv_buf, source=0)
            comm.Ssend(send_buf, dest=0)

    end = MPI.Wtime()
    return (end - start) / (2 * i)



for s in sizes:
    t = pingpong_ssend(s, N)
    if rank == 0:
        print(f"size {s}  time {t}")

lat = pingpong_ssend(sizes[0], N)

if rank == 0:
    print("\nlatency:", f"{lat}")