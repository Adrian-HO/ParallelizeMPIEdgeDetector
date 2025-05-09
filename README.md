# ParallelizeMPIEdgeDetector
to run, move code to wsl(ubuntu) environment
source ~/edge_detection_env/bin/activate
mpiexec -n 4 python ParallelCanny.py
deactivate