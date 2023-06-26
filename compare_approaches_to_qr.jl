using CUDA

# compare gelsBatched, geqrfBatched and CPU generation
# the GPU routines will probably require to use inv and multiply the inpu by R^-1 to obtain Q (getri, getrf)