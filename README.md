# tuning-competition-baseline

## インストール

1. 環境構築
```bash
cd /path/to/your/project_dir # 任意のディレクトリ
git clone https://github.com/llm-jp/tuning-competition-baseline.git
cd tuning-competition-baseline
bash ./scripts/mdx/install.sh
```

2.  

Please copy base_template.yaml to base.yaml and fix the values `work_dir` and `data_dir`.
`work_dir` is the directory where the model and the log files are stored and `data_dir` is the directory where the input data files are stored.

```bash
cp configs/base_template.yaml configs/base.yaml
```

## Checkpoint Conversion

### Hugging Face -> Nemo

`scripts/{mdx,sakura}/ckpt/convert_llama_hf_to_nemo.sh` converts the Hugging Face checkpoint to the Nemo checkpoint.
You need to modify `FIXME` in the script and run it.

```bash
# mdx
sbatch scripts/mdx/ckpt/convert_llama_hf_to_nemo.sh

# sakura
sbatch scripts/sakura/ckpt/convert_llama_hf_to_nemo.sh
```

### Nemo -> Hugging Face

`scripts/{mdx,sakura}/ckpt/convert_llama_nemo_to_hf.sh` converts the Nemo checkpoint to the Hugging Face checkpoint.
You need to modify `FIXME` in the script and run it.

```bash
# mdx
sbatch scripts/mdx/ckpt/convert_llama_nemo_to_hf.sh

# sakura
sbatch scripts/sakura/ckpt/convert_llama_nemo_to_hf.sh
```

## Training

Before training, you need to modify SLURM settings (``job-name``, ``output/error log file path``, ...) in the script.

Also, you need to modify the `FIXME` in the script as needed.

### mdx
Here are some examples of training scripts for mdx (llm-jp-nvlink cluster).
```bash
# 1.7B model with 2 nodes (16 GPUs)
sbatch scripts/mdx/mpi/1.7b.sh

# 13B model with 4 nodes (32 GPUs)
sbatch scripts/sakura/mpi/13b.sh

# 13B model with 8 nodes (64 GPUs)
sbatch --nodes 8 scripts/mdx/mpi/13b.sh
```


### sakura
```bash
# 1.7B model with 2 nodes (16 GPUs)
sbatch scripts/sakura/mpi/1.7b.sh

# 13B model with 4 nodes (32 GPUs)
sbatch scripts/sakura/mpi/13b.sh

# 172B model with 8 nodes (64 GPUs)
sbatch scripts/sakura/mpi/172b.sh
```
