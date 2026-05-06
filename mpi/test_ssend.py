#!/usr/bin/env python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 2:
    if rank == 0:
        print("Run with exactly 2 processes")
    raise SystemExit

if rank == 0:
    data = b"hello"
    print("rank 0: ssend -> rank 1")
    comm.ssend(data, dest=1, tag=0)
    print("rank 0: waiting for reply")
    reply = comm.recv(source=1, tag=1)
    print("rank 0: got reply:", reply)

elif rank == 1:
    print("rank 1: waiting")
    data = comm.recv(source=0, tag=0)
    print("rank 1: got:", data)
    comm.ssend(data, dest=0, tag=1)
    print("rank 1: reply sent")