#!/usr/bin/env python
from mpi4py import MPI
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def usage():
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(
            "Usage:\n"
            "  mpiexec -machinefile ./allnodes -np 2 ./pingpong_safe.py <mode> <min_size> <max_size> <iterations> <warmup>\n\n"
            "Modes:\n"
            "  ssend | rsend\n"
        )


def pingpong_ssend(comm, rank, msg, iterations, warmup):
    # Warmup
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


def pingpong_rsend(comm, rank, msg, iterations, warmup):
    """
    Ready send wymaga, by receive było już wystawione.
    Dla bezpieczeństwa robimy mały handshake przez zwykłe send/recv.
    To dodaje narzut, ale zapewnia poprawność semantyczną.
    """

    # Warmup
    for _ in range(warmup):
        if rank == 1:
            # rank 1 gotowy na odbiór
            comm.send("ready0", dest=0, tag=100)
            data = comm.recv(source=0, tag=0)

            # rank 1 gotowy odesłać, ale rank 0 najpierw wystawi recv
            token = comm.recv(source=0, tag=101)
            if token != "ready1":
                raise RuntimeError("Protocol error")
            comm.rsend(data, dest=0, tag=1)

        else:  # rank 0
            token = comm.recv(source=1, tag=100)
            if token != "ready0":
                raise RuntimeError("Protocol error")
            comm.rsend(msg, dest=1, tag=0)

            # wystaw recv przez recv() i daj znać rank 1, że może rsend
            req = comm.irecv(source=1, tag=1)
            comm.send("ready1", dest=1, tag=101)
            _ = req.wait()

    comm.Barrier()

    if rank == 0:
        t0 = MPI.Wtime()

    for _ in range(iterations):
        if rank == 1:
            comm.send("ready0", dest=0, tag=100)
            data = comm.recv(source=0, tag=0)

            token = comm.recv(source=0, tag=101)
            if token != "ready1":
                raise RuntimeError("Protocol error")
            comm.rsend(data, dest=0, tag=1)

        else:  # rank 0
            token = comm.recv(source=1, tag=100)
            if token != "ready0":
                raise RuntimeError("Protocol error")
            comm.rsend(msg, dest=1, tag=0)

            req = comm.irecv(source=1, tag=1)
            comm.send("ready1", dest=1, tag=101)
            _ = req.wait()

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
            print("Run with exactly 2 processes", file=sys.stderr, flush=True)
        raise SystemExit(1)

    if len(sys.argv) != 6:
        usage()
        raise SystemExit(1)

    mode = sys.argv[1].lower()
    min_size = int(sys.argv[2])
    max_size = int(sys.argv[3])
    iterations = int(sys.argv[4])
    warmup = int(sys.argv[5])

    if mode not in ("ssend", "rsend"):
        if rank == 0:
            print("Mode must be: ssend or rsend", file=sys.stderr, flush=True)
        raise SystemExit(1)

    if rank == 0:
        print("mode,message_size_B,iterations,avg_rtt_s,one_way_latency_s,bandwidth_Mbit_s")
        sys.stdout.flush()

    msg_size = min_size
    while msg_size <= max_size:
        if rank == 0:
            print("[INFO] testing size =", msg_size, "B", file=sys.stderr, flush=True)

        msg = b"x" * msg_size

        if mode == "ssend":
            avg_rtt = pingpong_ssend(comm, rank, msg, iterations, warmup)
        else:
            avg_rtt = pingpong_rsend(comm, rank, msg, iterations, warmup)

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