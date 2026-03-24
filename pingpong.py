#!/usr/bin/env python3
from mpi4py import MPI
import sys


def usage():
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(
            "Usage:\n"
            "  mpirun -np 2 python mpi_pingpong_measure.py <mode> <min_size> <max_size> <iterations> <warmup>\n\n"
            "Arguments:\n"
            "  mode        : ssend | rsend\n"
            "  min_size    : minimum message size in bytes (e.g. 1)\n"
            "  max_size    : maximum message size in bytes (e.g. 1048576)\n"
            "  iterations  : number of ping-pong iterations per size (e.g. 10000)\n"
            "  warmup      : number of warmup iterations per size (e.g. 100)\n"
            "Example:\n"
            "  mpirun -np 2 python mpi_pingpong_measure.py ssend 1 1048576 10000 100 > ssend.csv"
        )


def pingpong_ssend(comm, rank, buf, msg_size, iterations, warmup):
    peer = 1 - rank

    # Warmup
    for _ in range(warmup):
        if rank == 0:
            comm.Ssend([buf, msg_size, MPI.BYTE], dest=peer, tag=0)
            comm.Recv([buf, msg_size, MPI.BYTE], source=peer, tag=0)
        else:
            comm.Recv([buf, msg_size, MPI.BYTE], source=peer, tag=0)
            comm.Ssend([buf, msg_size, MPI.BYTE], dest=peer, tag=0)

    comm.Barrier()

    if rank == 0:
        t0 = MPI.Wtime()

    for _ in range(iterations):
        if rank == 0:
            comm.Ssend([buf, msg_size, MPI.BYTE], dest=peer, tag=0)
            comm.Recv([buf, msg_size, MPI.BYTE], source=peer, tag=0)
        else:
            comm.Recv([buf, msg_size, MPI.BYTE], source=peer, tag=0)
            comm.Ssend([buf, msg_size, MPI.BYTE], dest=peer, tag=0)

    if rank == 0:
        t1 = MPI.Wtime()
        return (t1 - t0) / iterations  # avg RTT

    return None


def pingpong_rsend(comm, rank, buf, msg_size, iterations, warmup):
    """
    Bezpieczny ping-pong z Rsend.
    Używamy dwóch Barrier na iterację, żeby zagwarantować:
    1) rank 1 wystawił Irecv przed Rsend 0->1
    2) rank 0 wystawił Irecv przed Rsend 1->0

    Uwaga: ten narzut wpływa na wynik, ale semantyka Rsend jest zachowana.
    """
    peer = 1 - rank

    # Warmup
    for _ in range(warmup):
        if rank == 1:
            # Przygotuj odbiór 0 -> 1
            req = comm.Irecv([buf, msg_size, MPI.BYTE], source=peer, tag=0)
            comm.Barrier()   # sygnał: Irecv(tag=0) gotowy
            req.Wait()

            # Przygotuj odbiór odpowiedzi 1 -> 0 po stronie 0
            comm.Barrier()   # czekamy aż rank 0 wystawi Irecv(tag=1)
            comm.Rsend([buf, msg_size, MPI.BYTE], dest=peer, tag=1)

        else:  # rank 0
            comm.Barrier()   # czekamy aż rank 1 wystawi Irecv(tag=0)
            comm.Rsend([buf, msg_size, MPI.BYTE], dest=peer, tag=0)

            req = comm.Irecv([buf, msg_size, MPI.BYTE], source=peer, tag=1)
            comm.Barrier()   # sygnał: Irecv(tag=1) gotowy
            req.Wait()

    comm.Barrier()

    if rank == 0:
        t0 = MPI.Wtime()

    for _ in range(iterations):
        if rank == 1:
            # Etap 1: przygotuj odbiór 0 -> 1
            req = comm.Irecv([buf, msg_size, MPI.BYTE], source=peer, tag=0)
            comm.Barrier()   # rank 0 może zrobić Rsend(tag=0)
            req.Wait()

            # Etap 2: czekaj aż rank 0 wystawi odbiór 1 -> 0
            comm.Barrier()
            comm.Rsend([buf, msg_size, MPI.BYTE], dest=peer, tag=1)

        else:  # rank 0
            # Etap 1: czekaj aż rank 1 wystawi Irecv(tag=0)
            comm.Barrier()
            comm.Rsend([buf, msg_size, MPI.BYTE], dest=peer, tag=0)

            # Etap 2: wystaw odbiór odpowiedzi 1 -> 0
            req = comm.Irecv([buf, msg_size, MPI.BYTE], source=peer, tag=1)
            comm.Barrier()   # rank 1 może teraz zrobić Rsend(tag=1)
            req.Wait()

    if rank == 0:
        t1 = MPI.Wtime()
        return (t1 - t0) / iterations  # avg RTT

    return None


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_size != 2:
        if rank == 0:
            print("This program requires exactly 2 MPI processes.", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) != 6:
        usage()
        sys.exit(1)

    mode = sys.argv[1].lower()
    min_size = int(sys.argv[2])
    max_size = int(sys.argv[3])
    iterations = int(sys.argv[4])
    warmup = int(sys.argv[5])

    if mode not in ("ssend", "rsend"):
        if rank == 0:
            print("Mode must be: ssend or rsend", file=sys.stderr)
        sys.exit(1)

    if min_size <= 0 or max_size < min_size or iterations <= 0 or warmup < 0:
        if rank == 0:
            print("Invalid arguments.", file=sys.stderr)
        sys.exit(1)

    buf = bytearray(max_size)

    if rank == 0:
        print("mode,message_size_B,iterations,avg_rtt_s,one_way_latency_s,bandwidth_Mbit_s")

    msg_size = min_size
    while msg_size <= max_size:
        if mode == "ssend":
            avg_rtt = pingpong_ssend(comm, rank, buf, msg_size, iterations, warmup)
        else:
            avg_rtt = pingpong_rsend(comm, rank, buf, msg_size, iterations, warmup)

        if rank == 0:
            one_way_latency = avg_rtt / 2.0
            bandwidth_Bps = (2.0 * msg_size) / avg_rtt
            bandwidth_Mbit_s = (bandwidth_Bps * 8.0) / 1e6

            print(
                f"{mode},{msg_size},{iterations},"
                f"{avg_rtt:.12e},{one_way_latency:.12e},{bandwidth_Mbit_s:.6f}"
            )
            sys.stdout.flush()

        msg_size *= 2


if __name__ == "__main__":
    main()