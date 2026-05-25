RAY_PORT=2468
RAY_HEAD_IP=<YOUR_RAY_HEAD_IP>
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_SOCKET_IFNAME=ens1f0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 ray start --head --port=$RAY_PORT --resources='{"docker:'$RAY_HEAD_IP'": 128}'
docker stop $(docker ps -a -q)