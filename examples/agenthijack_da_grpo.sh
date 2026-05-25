set -x
# RAY_PORT=2468
# RAY_HEAD_IP=<YOUR_RAY_HEAD_IP>
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
export WANDB_KEY="<YOUR_WANDB_API_KEY>"
export SWANLAB_API_KEY="<YOUR_SWANLAB_API_KEY>"
# export NCCL_SOCKET_IFNAME=ens34
# export NCCL_SOCKET_IFNAME=^docker0,ens1f1,ens1f2,ens2f0,ens2f1,lo
export NCCL_SOCKET_IFNAME=ens1f0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
# export SWANLAB_RESUME=must
# export SWANLAB_RUN_ID=<YOUR_SWANLAB_RUN_ID>  # set to resume a previous run

MODEL_PATH=<PATH_TO_UI-TARS-1.5-7B_OR_YOUR_CHECKPOINT>

SYSTEM_PROMPT="""You are helpful assistant."""

NUM_GPUS=4
NUM_ENVS=4
ROLLOUT_N=4

((ROLLOUT_BSZ = NUM_ENVS/ROLLOUT_N))

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.format_prompt="${SYSTEM_PROMPT}" \
    data.train_files=AgentHijack/evaluation_examples/test_success_uitars1.5_wo_impossible.json \
    data.val_files=AgentHijack/evaluation_examples/test_success_uitars1.5_wo_impossible.json \
    data.max_prompt_length=64000 \
    data.max_response_length=8192 \
    data.max_pixels=2116800 \
    data.min_pixels=256 \
    data.rollout_batch_size=1 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.max_grad_norm=1.0 \
    worker.actor.optim.lr=1e-6 \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.actor.ulysses_sequence_parallel_size=1 \
    worker.actor.padding_free=true \
    worker.actor.ppo_epochs=1 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.3 \
    worker.actor.global_batch_size=1 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.rollout.temperature=1.0 \
    worker.rollout.n=$ROLLOUT_N \
    worker.rollout.limit_images=15 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.max_num_batched_tokens=128000 \
    algorithm.disable_kl=True \
    algorithm.kl_coef=0 \
    algorithm.enable_replay=True \
    env.num_envs=$NUM_ENVS \
    env.max_steps=10 \
    trainer.experiment_name=osworld_cot_7b_nokl_subset32_ours \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=128 \
    trainer.save_limit=3 \
    trainer.val_before_train=False \
    trainer.val_freq=128 \
    trainer.total_episodes=12
