#!/usr/bin/env python
from mpi4py import MPI
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


def usage():
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(
            "Usage:\n"
            "  mpiexec -machinefile ./MPR/allnodes -np 2 ./MPR/pingpong.py <mode> <min_size> <max_size> <iterations> <warmup>\n\n"
            "Modes:\n"
            "  send | ssend\n",
            flush=True
        )


def pingpong_send(comm, rank, msg, iterations, warmup):
    # Warmup (not measured)
    for _ in range(warmup):
        if rank == 0:
            comm.send(msg, dest=1, tag=0)
            _ = comm.recv(source=1, tag=1)
        else:
            data = comm.recv(source=0, tag=0)
            comm.send(data, dest=0, tag=1)

    comm.Barrier()

    if rank == 0:
        t0 = MPI.Wtime()

    for _ in range(iterations):
        if rank == 0:
            comm.send(msg, dest=1, tag=0)
            _ = comm.recv(source=1, tag=1)
        else:
            data = comm.recv(source=0, tag=0)
            comm.send(data, dest=0, tag=1)

    if rank == 0:
        t1 = MPI.Wtime()
        return (t1 - t0) / iterations  # avg RTT
    return None


def pingpong_ssend(comm, rank, msg, iterations, warmup):
    # Warmup (not measured)
    for _ in range(warmup):
        if rank == 0:
            comm.ssend(msg, dest=1, tag=0)
            _ = comm.recv(source=1, tag=1)
        else:
            data = comm.recv(source=0, tag=0)
            comm.ssend(data, dest=0, tag=1)

    comm.Barrier()

    if rank == 0:
        t0 = MPI.Wtime()

    for _ in range(iterations):
        if rank == 0:
            comm.ssend(msg, dest=1, tag=0)
            _ = comm.recv(source=1, tag=1)
        else:
            data = comm.recv(source=0, tag=0)
            comm.ssend(data, dest=0, tag=1)

    if rank == 0:
        t1 = MPI.Wtime()
        return (t1 - t0) / iterations  # avg RTT
    return None


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 2:
        if rank == 0:
            eprint("[ERROR] Run with exactly 2 processes")
        raise SystemExit(1)

    if len(sys.argv) != 6:
        usage()
        raise SystemExit(1)

    mode = sys.argv[1].lower()
    min_size = int(sys.argv[2])
    max_size = int(sys.argv[3])
    iterations = int(sys.argv[4])
    warmup = int(sys.argv[5])

    if mode not in ("send", "ssend"):
        if rank == 0:
            eprint("[ERROR] Mode must be: send or ssend")
        raise SystemExit(1)

    if rank == 0:
        print("mode,message_size_B,iterations,avg_rtt_s,one_way_latency_s,bandwidth_Mbit_s")

    msg_size = min_size
    while msg_size <= max_size:
        if rank == 0:
            print("[INFO] testing size =", msg_size, "B", file=sys.stderr, flush=True)

        msg = b"x" * msg_size

        if mode == "send":
            avg_rtt = pingpong_send(comm, rank, msg, iterations, warmup)
        else:
            avg_rtt = pingpong_ssend(comm, rank, msg, iterations, warmup)

        if rank == 0:
            one_way_latency = avg_rtt / 2.0
            bandwidth_Bps = (2.0 * msg_size) / avg_rtt
            bandwidth_Mbit_s = (bandwidth_Bps * 8.0) / 1e6

            print(
                f"{mode},{msg_size},{iterations},"
                f"{avg_rtt:.12e},{one_way_latency:.12e},{bandwidth_Mbit_s:.6f}",
                flush=True
            )

        msg_size *= 2


if __name__ == "__main__":
    main()