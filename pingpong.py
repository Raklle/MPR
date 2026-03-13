#!/usr/bin/env python3
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


N = 100000
sizes = [1, 8, 16]


def pingpong_rsend(msg_size, iterations):
    send_buf = bytearray(msg_size)
    recv_buf = bytearray(msg_size)
    sync = bytearray(1)

    comm.Barrier()
    start = MPI.Wtime()

    for _ in range(iterations):
        if rank == 0:
            req = comm.Irecv(recv_buf, source=1)
            comm.Send(sync, dest=1)
            comm.Recv(sync, source=1)
            comm.Rsend(send_buf, dest=1)
            req.Wait()
        else:
            req = comm.Irecv(recv_buf, source=0)
            comm.Recv(sync, source=0)
            comm.Send(sync, dest=0)
            req.Wait()
            comm.Rsend(send_buf, dest=0)

    end = MPI.Wtime()
    return (end - start) / (2 * iterations)



for s in sizes:
    t = pingpong_rsend(s, N)
    if rank == 0:
        print(f"size:{s:9d}  time:{t:.9e}")

lat = pingpong_rsend(sizes[0], N)

if rank == 0:
    print("\ndelay:", f"{lat:.9e}", "s")