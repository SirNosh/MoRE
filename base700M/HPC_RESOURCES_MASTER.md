# HPC Resources Master Documentation
**User:** dvyas4  
**Cluster:** ACIDSDB (Arctic)  
**Last Updated:** 2025-10-26

---

## 📋 Quick Reference: Best GPU Nodes for Deep Learning

| Node | GPUs | GPU Memory | System RAM | Cores | Partition | Best For |
|------|------|------------|------------|-------|-----------|----------|
| **acidsgcn007** | 8x V100 32GB | 256GB total | 768GB | 32 | qGPU* | **Multi-GPU training** |
| **acidsgcn001-006** | 4x V100 32GB | 128GB total | 384GB | 40 | qGPU* | **Large models** |
| **acidsgcn008** | 3x A30 24GB | 72GB total | 256GB | 128 | qGPU* | **Memory-efficient training** |
| acidsgcn011-012 | 8x V100 32GB | 256GB total | TBD | TBD | qGPU* | Multi-GPU alternative |

---

## 🎯 Accessible Partitions

### GPU Partitions (Primary for ML/DL)

#### **qGPU24** (Recommended for this project)
- **Time Limit:** 1 day (24 hours)
- **Status:** ✅ ACCESSIBLE
- **GPU Nodes:**
  - `acidsgcn001-006`: 4x V100 SXM2 32GB each (384GB RAM, 40 cores)
  - `acidsgcn007`: 8x V100 SXM2 32GB (768GB RAM, 32 cores) **[BEST OPTION]**
  - `acidsgcn008`: 3x A30 24GB (256GB RAM, 128 cores)
  - `acidsgcn010`: 6x L40S (specific config TBD)
  - `acidsgcn011-012`: 8x V100 SXM2 32GB each
  - Single GPU nodes: acidsgn001-009 (RTX 2080 Ti 12GB or TITAN V 12GB)

#### **qGPU48**
- **Time Limit:** 2 days (48 hours)
- **Status:** ✅ ACCESSIBLE
- **GPU Nodes:** Same as qGPU24
- **Use Case:** Longer training jobs

#### **qGPU120**
- **Time Limit:** 5 days (120 hours)
- **Status:** ✅ ACCESSIBLE
- **GPU Nodes:** Same as qGPU24
- **Use Case:** Extended training experiments

### Development Partition

#### **qDEV**
- **Time Limit:** 12 hours
- **Status:** ✅ ACCESSIBLE
- **Use Case:** Testing, debugging, short experiments
- **GPU Nodes Available:**
  - `acidsgcn007`: 8x V100 32GB
  - `acidsmn003`: 2x V100 PCIe 16GB
  - `arcdevelopment`: 1x V100 PCIe 32GB
  - Single GPU nodes with RTX 2080 Ti

### CPU/Memory Partitions

#### **qCPU24/48/120**
- **Time Limits:** 1 day / 2 days / 5 days
- **Status:** ✅ ACCESSIBLE
- **Use Case:** Data preprocessing, CPU-intensive tasks
- **Resources:** 24-128 cores, 192GB-256GB RAM per node

#### **qMEM24/48/120**
- **Time Limits:** 1 day / 2 days / 5 days
- **Status:** ✅ ACCESSIBLE
- **Use Case:** High-memory jobs (up to 1.5TB)
- **Notable Nodes:**
  - `acidsmn001`: 1.5TB RAM
  - `acidsmn002`: 3TB RAM
  - `acidscn002, acidscn006`: 1TB RAM

---

## ❌ Non-Accessible Partitions

### **qTRDGPU** ⛔
- **Status:** ❌ NOT ACCESSIBLE (permission denied)
- **Contains:** NVIDIA A40 (48GB), RTX 2080 Ti nodes
- **Error:** "User's group not permitted to use this partition"

### **qTRDGPUH** ⛔
- **Status:** ❌ NOT ACCESSIBLE
- **Contains:** 
  - 8x A100 80GB nodes (arctrddgxa001) - Best hardware, but blocked
  - 8x V100 32GB nodes (arctrddgx001-004)
- **Note:** Premium hardware reserved for specific research groups

### **qTRDBF, qTRDGPUBF, qTRDBD, pwrTest** ⛔
- **Status:** ❌ NOT ACCESSIBLE
- Special purpose partitions

---

## 🖥️ Detailed Node Specifications

### Multi-GPU V100 Nodes (BEST FOR YOUR PROJECT)

#### **acidsgcn007** ⭐ PRIMARY RECOMMENDATION
```
GPUs:       8x Tesla V100 SXM2 32GB (256GB GPU memory total)
System RAM: 768GB
CPU Cores:  32
Network:    25G Ethernet, 100G InfiniBand
Partitions: qDEV, qGPU24, qGPU48, qGPU120
GRES:       gpu:V100:8
```
**Best for:** Full neuroplasticity experiments with all 4 models

#### **acidsgcn001-006** ⭐ BACKUP OPTION
```
GPUs:       4x Tesla V100 SXM2 32GB (128GB GPU memory total)
System RAM: 384GB
CPU Cores:  40
Network:    10-25G Ethernet, 100G InfiniBand
Partitions: qGPU24, qGPU48, qGPU120
GRES:       gpu:V100:4
```
**Best for:** Running 2 experiments in parallel or large single models

#### **acidsgcn008** (Alternative)
```
GPUs:       3x NVIDIA A30 24GB (72GB GPU memory total)
System RAM: 256GB
CPU Cores:  128
Network:    10G Ethernet, 100G InfiniBand
Partitions: qGPU24, qGPU48, qGPU120
GRES:       gpu:A30:3
```
**Best for:** Memory-efficient training, highly parallel CPU tasks

### Single GPU Nodes

#### Development/Testing Nodes
```
acidsgn001: TITAN V 12GB (256GB RAM, 64 cores) - qDEV, qCPU*
acidsgn002-009: RTX 2080 Ti 12GB (256GB RAM, 64-128 cores) - qDEV, qCPU*
acidsgn007: RTX 2080 Ti 12GB (1TB RAM, 64 cores) - qDEV, qCPU*, qMEM*
```

### Special Purpose Nodes

#### **acidsmn003** (High Memory + GPU)
```
GPUs:       2x V100 PCIe 16GB
System RAM: 1TB (high memory)
CPU Cores:  64
Partitions: qDEV, qMEM24, qMEM48, qMEM120
Use Case:   Large dataset loading + GPU training
```

---

## 📊 Resource Allocation Strategy

### For Continual Learning Project (Qwen2.5-7B)

#### **Strategy 1: Single Large Node (RECOMMENDED)**
```bash
Partition:  qGPU48 or qGPU120
Node:       acidsgcn007 (8x V100 32GB)
Allocation: --gres=gpu:V100:1 (use 1 of 8 GPUs)
Time:       24 hours (sufficient for all 4 experiments)
Memory:     128GB (plenty for 7B model with 4-bit quantization)
CPUs:       16 cores
```
**Advantages:**
- Most powerful single GPU (V100 32GB)
- Plenty of system RAM (768GB)
- No OOM issues
- Can run all experiments sequentially

#### **Strategy 2: Parallel Execution**
```bash
Partition:  qGPU48
Node:       acidsgcn001-006 (4x V100 32GB)
Jobs:       Submit 4 separate jobs, each using 1 GPU
Time:       6 hours per job (24 hours total wall time)
```
**Advantages:**
- All experiments complete in ~6 hours (parallel)
- More efficient use of cluster time

#### **Strategy 3: Development Testing**
```bash
Partition:  qDEV
Node:       Any available GPU node
Time:       Up to 12 hours
Use:        Test scripts before long production runs
```

---

## 🔧 SLURM Submission Examples

### Request Single V100 GPU
```bash
#SBATCH --partition=qGPU120
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
```

### Request Multiple V100 GPUs (for parallel jobs)
```bash
#SBATCH --partition=qGPU120
#SBATCH --gres=gpu:V100:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
```

### Request A30 GPU (memory efficient)
```bash
#SBATCH --partition=qGPU48
#SBATCH --gres=gpu:A30:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
```

---

## ⚠️ Important Notes

### Time Limits
- **qGPU24:** 24 hours maximum
- **qGPU48:** 48 hours maximum
- **qGPU120:** 120 hours (5 days) maximum
- **qDEV:** 12 hours maximum (for testing only)

### Memory Considerations
- V100 32GB: Ideal for 7B models with 4-bit quantization
- V100 16GB: May require more aggressive memory optimization
- RTX 2080 Ti 12GB: Tight but workable with batch_size=1

### Best Practices
1. **Always test in qDEV first** before submitting long jobs
2. **Use qGPU48 or qGPU120** for production runs (allows 2-5 days)
3. **Request single GPU unless truly need multiple** (more likely to schedule quickly)
4. **Set realistic time limits** (job killed if exceeds limit)
5. **Monitor jobs:** `squeue -u dvyas4`

---

## 📂 File Paths

### Home Directory
```
/home/users/dvyas4/
```

### Project Directory
```
/home/users/dvyas4/Neuroplastic-COT/
```

### Scratch/Temp (if needed)
```
/tmp/                    # Local node scratch
```

---

## 🚀 Recommended Configuration for Your Project

**Partition:** `qGPU48` (2-day limit, sufficient for all experiments)  
**Node Type:** `acidsgcn007` or `acidsgcn001-006` (V100 nodes)  
**GPU Request:** `--gres=gpu:V100:1` (single V100 32GB)  
**Time:** `24:00:00` (24 hours, plenty for 4 sequential experiments)  
**Memory:** `128G` (more than enough)  
**CPUs:** `16` (good for data loading parallelism)

**Why this config:**
- V100 32GB handles 7B model with 4-bit quantization comfortably
- 24 hours sufficient for all 4 training runs (baseline, EWC, SI, full)
- 128GB RAM prevents any memory issues during Fisher computation
- Single GPU request = faster scheduling

---

## 📞 Support Resources

### Check Available Resources
```bash
# View partition status
sinfo

# Check your jobs
squeue -u dvyas4

# Check specific partition
scontrol show partition qGPU120

# View node details
scontrol show node acidsgcn007
```

### Monitor Running Jobs
```bash
# View output live
tail -f job-output-<JOBID>.out

# Check GPU usage on node
ssh <nodename>
nvidia-smi
```

### Cancel Jobs
```bash
scancel <JOBID>
```

---

## 📝 Version History

- **2025-10-26:** Initial documentation
  - Verified partition access (qGPU*, qDEV, qCPU*, qMEM* accessible)
  - Documented qTRDGPU* partitions are NOT accessible
  - Identified acidsgcn007 as best node for project

---

**END OF MASTER DOCUMENTATION**
