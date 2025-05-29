# Distributed Training

This guide focuses on setting up and running distributed training across multiple compute nodes. It outlines the necessary prerequisites, key configuration parameters to adjust, and how to launch a multi-node job.

## Prerequisites for Multi-Node Training

Effective multi-node training relies on seamless communication between the nodes.

**Password-less SSH:**

1.  On the master node, generate an SSH key pair if you don't have one: `ssh-keygen -t rsa` (or a more modern algorithm like ed25519).

2.  Copy the public key from the master node to the `~/.ssh/authorized_keys` file on each worker node. You can use `ssh-copy-id user@worker_node_ip`.

3.  Ensure the `~/.ssh` directory on worker nodes has `700` permissions and `~/.ssh/authorized_keys` has `600` permissions.

4.  Test by trying to SSH from the master node to each worker node: `ssh user@worker_node_ip`. It should connect without asking for a password.

## Configuration for Multi-Node Training

When scaling to multiple nodes, you'll primarily adjust parallelism settings to distribute the model and data effectively. The total number of GPUs you have (`N_total_GPUs`) will be utilized by a combination of these strategies:

`N_total_GPUs = (PipelineParallelism_Size * TensorParallelism_Size) * DataParallelism_Size`

* **`system.pipeline_model_parallel_size` (PP):**

    * **Role in Multi-Node:** This is crucial for inter-node parallelism. You typically set this to the number of nodes, or a multiple, to distribute stages of your model across different nodes.

    * **How to Modify:**

        * Default in provided config: `1`

        * For N nodes, you might set `pipeline_model_parallel_size: N`. For example, if you have 2 nodes, setting it to `2` would place one pipeline stage on each node (assuming TP and other parallelisms are handled within the node or stage).

        * If a node has multiple GPUs, each pipeline stage can itself be parallelized using Tensor Parallelism on those GPUs.

* **`system.tensor_model_parallel_size` (TP):**

    * **Role in Multi-Node:** Primarily for intra-node parallelism. It splits individual model layers across GPUs *within* a node (or within a pipeline stage).

    * **How to Modify:**

        * Default in provided config: `1`

        * Set this to the number of GPUs per node (or per pipeline stage) you want to dedicate to tensor parallelism. E.g., if a node has 8 GPUs, you might use `tensor_model_parallel_size: 4` or `8`. This requires high-bandwidth interconnects (like NVLink) between these GPUs.

* **`system.expert_model_parallel_size` (EP) (for MoE models):**

    * **Role in Multi-Node:** Usually applied intra-node, often within a tensor-parallel group. If `model.num_experts` is large (e.g., `64` in the config), you'll distribute these experts across a subset of GPUs.

    * **How to Modify:**

        * Default in provided config: `1`

        * If `tensor_model_parallel_size` is, for example, `4`, you might also set `expert_model_parallel_size: 4` so that the 4 GPUs in the TP group also handle a fraction of the experts.

        * The `model.moe_token_dispatcher_type: "alltoall"` is typically used for this.

* **`system.context_parallel_size` (CP):**

    * **Role in Multi-Node:** Can be used in conjunction with TP to reduce activation memory for very long sequences. Its application in multi-node setups depends on how it interacts with TP and PP in your framework.

    * **How to Modify:**

        * Default in provided config: `1`

        * Consider increasing if activation memory for long sequences is a bottleneck after applying TP and PP.

* **Data Parallelism (DP) Size:**

    * **Role in Multi-Node:** This is often derived. After setting TP and PP, the remaining degree of parallelism across your total GPUs becomes your data parallelism.

    * `DataParallelism_Size = N_total_GPUs / (pipeline_model_parallel_size * tensor_model_parallel_size)`

    * Each data parallel replica processes a different batch of data.

* **Batch Sizes:**

    * **`model.global_batch_size`**: (e.g., `1024` in config)

        * This is your target effective batch size across all operations. Keep this consistent with your experimental setup.

    * **`model.micro_batch_size`**: (e.g., `1` in config)

        * The batch size processed by a single GPU (or pipeline stage) in one forward/backward pass. This is critical for managing memory on each GPU.

        * Adjust this to be as large as possible without OOM errors on any GPU.

        * Gradient Accumulation Steps (GAS) will be determined by:
            `GAS = model.global_batch_size / (model.micro_batch_size * DataParallelism_Size)`

* **Important Flags to Keep Enabled:**

    * `system.sequence_parallel: true` (if TP > 1)

    * `system.use_distributed_optimizer: true` (critical for memory with large models)

    * `system.overlap_grad_reduce: true`

    * `system.overlap_param_gather: true`

## Example

Let's consider a setup with **2 nodes, each having 8 GPUs (total 16 GPUs)**. Your goal is to train a large model.

**Configuration Approach (PP across nodes, TP within nodes):**

* `system.pipeline_model_parallel_size: 2` (One pipeline stage per node)

* `system.tensor_model_parallel_size: 8` (All 8 GPUs on each node used for tensor parallelism within their pipeline stage)

* `system.expert_model_parallel_size`: If using MoE, this could be `8` as well, aligned with TP.

* **Data Parallelism Size**: `16 / (2 * 8) = 1`. In this setup, there's only one data parallel replica of the entire pipelined model. To increase data parallelism, you'd need to reduce TP or PP, or add more nodes/GPUs.

* **Alternative with DP**: If you wanted DP=2:
    * `system.pipeline_model_parallel_size: 2`

    * `system.tensor_model_parallel_size: 4`

    * Data Parallelism Size: `16 / (2 * 4) = 2`. Now you have two data parallel replicas, each spanning 2 nodes with 4-way TP per stage.

* `model.micro_batch_size`: Adjust to fit GPU memory. If `1`, GAS = `1024 / (1 * DP_Size)`.

## Launching a Multi-Node Training Job

After modifying the configuration file according to the above instructions, start directly using the [script](openseek/algorithm/run_exp.sh).

```sh
bash openseek/algorithm/run_exp.sh start <config-path>
```

## Important Considerations

* **Network Bandwidth:** Inter-node communication is slower than intra-node (NVLink). Pipeline parallelism is often preferred for inter-node communication as it can be more latency-tolerant than trying to do fine-grained tensor parallelism across nodes.

* **Experimentation:** Finding the optimal TP, PP, and DP configuration for your specific model, hardware, and network requires experimentation. Profile your runs to identify bottlenecks.

