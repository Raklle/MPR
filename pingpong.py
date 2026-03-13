from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

N = 10000
msg = bytearray(1)

comm.Barrier()
start = MPI.Wtime()

for i in range(N):
    if rank == 0:
        comm.Recv(msg, source=1)
    elif rank == 1:
        comm.Recv(msg, source=0)
        comm.Send(msg, dest=0)

comm.Barrier()
end = MPI.Wtime()

if rank == 0:
    latency = (end - start) / (2 * N)
    print("Latency:", latency)