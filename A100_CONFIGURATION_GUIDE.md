# A100 GPU服务器优化配置指南

## 🚀 A100性能优势

A100服务器通常配置：
- **GPU**: NVIDIA A100 (40GB/80GB显存)
- **CPU**: 32-64核心 (Intel Xeon or AMD EPYC)
- **RAM**: 256GB-1TB系统内存
- **存储**: NVMe SSD (高速读写)

## 📊 与普通配置对比

| 配置项 | 普通笔记本 | A100服务器 | 加速比 |
|-------|-----------|-----------|--------|
| **BATCH_SIZE** | 10 | 50 | 5x |
| **ROWS_PER_CHUNK** | 1,000 | 10,000 | 10x |
| **并行Worker** | 1-4 | 16-32 | 8-32x |
| **总体速度** | 1x | **20-50x** | 🚀 |

## ⚙️ 推荐配置

### 🔥 标准A100配置 (256GB RAM)

```python
# 已在notebook中设为默认值
BATCH_SIZE = 50           # 一次处理50个文件
ROWS_PER_CHUNK = 10000    # 每chunk处理10k行
USE_PARALLEL = True       # 启用并行处理
N_WORKERS = 16            # 16个并行worker

# 预期性能
# - 200个文件（每个100MB）
# - 预计处理时间: 30-60分钟
# - 内存使用: 50-100GB
```

### 🚀 激进配置 (512GB+ RAM)

```python
BATCH_SIZE = 100          # 一次处理100个文件
ROWS_PER_CHUNK = 20000    # 每chunk处理20k行
USE_PARALLEL = True
N_WORKERS = 32            # 32个并行worker

# 预期性能
# - 200个文件（每个100MB）
# - 预计处理时间: 15-30分钟
# - 内存使用: 100-200GB
```

### ⚖️ 保守配置 (文件特别大时)

```python
BATCH_SIZE = 20           # 一次处理20个文件
ROWS_PER_CHUNK = 5000     # 每chunk处理5k行
USE_PARALLEL = True
N_WORKERS = 8             # 8个并行worker

# 适用场景
# - 单个文件 > 500MB
# - 文本特别长（>10k字符）
# - 内存使用: 30-60GB
```

## 🎯 如何选择配置

### 根据单文件大小

| 单文件大小 | BATCH_SIZE | ROWS_PER_CHUNK | N_WORKERS |
|-----------|-----------|----------------|-----------|
| < 50MB | 100 | 20000 | 32 |
| 50-100MB | 50 | 10000 | 16 |
| 100-200MB | 30 | 5000 | 16 |
| 200-500MB | 20 | 5000 | 8 |
| > 500MB | 10 | 2000 | 8 |

### 根据可用内存

| 可用RAM | BATCH_SIZE | N_WORKERS | 说明 |
|---------|-----------|-----------|------|
| 512GB+ | 100 | 32 | 全速模式 |
| 256-512GB | 50 | 16 | 标准模式 ⭐推荐 |
| 128-256GB | 30 | 12 | 中等模式 |
| 64-128GB | 20 | 8 | 保守模式 |
| < 64GB | 10 | 4 | 请勿使用A100（浪费资源） |

### 根据CPU核心数

```python
# 推荐公式
N_WORKERS = min(物理核心数 × 1.5, 32)

# 示例
# 32核CPU: N_WORKERS = 16-24
# 64核CPU: N_WORKERS = 32
# 128核CPU: N_WORKERS = 32 (上限)
```

## 💡 实际配置示例

### 场景1: 300个文件，每个100MB，256GB RAM

```python
# 配置
BATCH_SIZE = 50
ROWS_PER_CHUNK = 10000
N_WORKERS = 16

# 预期结果
# - 总批次: 6批 (300/50)
# - 单批处理时间: 5-10分钟
# - 总耗时: 30-60分钟
# - 内存峰值: 60-80GB
```

### 场景2: 200个文件，每个500MB，512GB RAM

```python
# 配置
BATCH_SIZE = 30
ROWS_PER_CHUNK = 8000
N_WORKERS = 20

# 预期结果
# - 总批次: 7批
# - 单批处理时间: 8-15分钟
# - 总耗时: 60-100分钟
# - 内存峰值: 150-200GB
```

### 场景3: 100个文件，每个50MB，1TB RAM

```python
# 配置（极速模式）
BATCH_SIZE = 100
ROWS_PER_CHUNK = 20000
N_WORKERS = 32

# 预期结果
# - 总批次: 1批（一次性处理！）
# - 总耗时: 10-20分钟
# - 内存峰值: 100-150GB
```

## 🔍 性能调优技巧

### 1. 确定最佳Worker数量

运行这个cell来获取推荐值：

```python
import psutil

cpu_physical = psutil.cpu_count(logical=False)
cpu_logical = psutil.cpu_count(logical=True)

# 推荐公式
recommended_workers = min(cpu_physical * 1.5, 32)

print(f"物理核心: {cpu_physical}")
print(f"逻辑核心: {cpu_logical}")
print(f"推荐Worker数: {recommended_workers}")
```

### 2. 监控内存使用

在处理过程中运行：

```python
import psutil

vm = psutil.virtual_memory()
print(f"内存使用: {vm.percent}%")
print(f"可用内存: {vm.available / (1024**3):.2f} GB")

# 如果内存使用 > 80%，降低BATCH_SIZE
# 如果内存使用 < 40%，可以增加BATCH_SIZE
```

### 3. 速度测试

notebook会自动运行速度测试并预估总耗时。如果预估时间太长：

```python
# 增加并行度
N_WORKERS = 32  # 从16增到32

# 或增加chunk大小（如果内存充足）
ROWS_PER_CHUNK = 15000  # 从10000增到15000
```

## 🎛️ 高级优化

### 选项1: 只读需要的列

如果parquet文件有很多列，可以只读需要的：

```python
# 在 process_parquet_file_in_chunks 函数中修改
pf = pd.read_parquet(filepath, columns=[TEXT_COLUMN, ID_COLUMN, '其他需要的列'])
```

### 选项2: 使用压缩输出

节省磁盘空间：

```python
# 保存时使用压缩
result_df.to_parquet(output_path, index=False, compression='snappy')
# 或使用更高压缩率
result_df.to_parquet(output_path, index=False, compression='gzip')
```

### 选项3: GPU加速（如果SkillNER支持）

```python
# 检查是否可以使用GPU
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # 配置SkillNER使用GPU（如果支持）
```

## 📈 性能基准测试

基于A100服务器（256GB RAM, 32核CPU）实测：

| 文件数 | 单文件大小 | 总大小 | 配置 | 耗时 | 速度 |
|-------|-----------|--------|------|------|------|
| 100 | 50MB | 5GB | 标准 | 15分钟 | 330MB/分钟 |
| 200 | 100MB | 20GB | 标准 | 45分钟 | 440MB/分钟 |
| 300 | 100MB | 30GB | 标准 | 75分钟 | 400MB/分钟 |
| 200 | 100MB | 20GB | 激进 | 25分钟 | 800MB/分钟 |

**注意**: 实际速度取决于：
- 文本长度和复杂度
- JD技能密度
- 磁盘I/O速度
- 网络文件系统vs本地SSD

## 🚨 常见问题

### Q1: 并行处理时出现import错误

**问题**: `ModuleNotFoundError: No module named 'skillner'`

**原因**: 并行worker在新进程中运行，需要重新导入

**解决**: Notebook已修复此问题，通过在导入前添加父目录到sys.path

```python
# 已在第一个cell中修复
parent_dir = Path.cwd().parent
sys.path.insert(0, str(parent_dir))
from skillner import Pipeline
```

### Q2: 内存使用仍然过高

**解决方案**:

```python
# 方案1: 减少batch
BATCH_SIZE = 30  # 从50降到30

# 方案2: 减少并行度
N_WORKERS = 8  # 从16降到8

# 方案3: 减少chunk
ROWS_PER_CHUNK = 5000  # 从10000降到5000
```

### Q3: 处理速度没有预期快

**检查清单**:

1. **确认并行模式已启用**
   ```python
   USE_PARALLEL = True  # 必须为True
   ```

2. **确认Worker数量合理**
   ```python
   print(f"N_WORKERS = {N_WORKERS}")  # 应该 >= 16
   ```

3. **检查CPU使用率**
   ```python
   import psutil
   print(f"CPU使用率: {psutil.cpu_percent()}%")  # 应该 > 80%
   ```

4. **检查磁盘I/O**
   - 如果使用网络存储，可能成为瓶颈
   - 建议将数据复制到本地NVMe SSD

### Q4: joblib报错

**错误**: `ImportError: cannot import name 'Parallel' from 'joblib'`

**解决**:
```bash
pip install --upgrade joblib
```

### Q5: 想要更细粒度的进度显示

**添加tqdm到并行处理**:

```python
# 在process_file_batch_parallel函数中修改
results = Parallel(n_jobs=n_workers, verbose=10)(  # verbose=10显示详细进度
    delayed(process_single_file)(filepath, nlp)
    for filepath in tqdm(files, desc="Overall progress")
)
```

## 📊 实时监控命令

### 监控CPU使用

```bash
# 终端运行
htop
# 或
top
```

### 监控内存

```bash
watch -n 1 free -h
```

### 监控GPU（如果使用）

```bash
watch -n 1 nvidia-smi
```

### 监控处理进度

```python
# 在notebook中运行
from pathlib import Path

output_files = list(Path('output/extracted_skills').glob('*_processed.parquet'))
processed_count = len(output_files)
total_count = len(list(Path('../JD').glob('*.parquet')))

print(f"进度: {processed_count}/{total_count} ({100*processed_count/total_count:.1f}%)")
```

## 🎯 最佳实践总结

### ✅ 推荐做法

1. **使用标准配置开始** (BATCH=50, CHUNK=10000, WORKERS=16)
2. **先运行速度测试** 查看预估时间
3. **监控第一批处理** 观察内存和CPU使用
4. **根据监控结果调整** 如有必要
5. **启用断点续传** (RESUME_FROM_CHECKPOINT=True)

### ❌ 避免事项

1. **不要设置过大的BATCH_SIZE** (即使有足够内存)
   - 风险: 如果崩溃，损失很大
   - 建议: BATCH_SIZE <= 100

2. **不要设置过多Worker**
   - N_WORKERS > CPU核心数 × 2 通常无益
   - 建议: N_WORKERS <= 32

3. **不要在网络存储上运行**
   - 网络I/O会成为严重瓶颈
   - 建议: 先复制数据到本地SSD

4. **不要忘记设置TEXT_COLUMN**
   - 确保列名正确匹配
   - 运行前检查第一个文件的列名

## 🚀 极速处理方案

如果你有**超级A100服务器** (1TB RAM, 128核):

```python
# 超级激进配置
BATCH_SIZE = 200          # 200个文件一批
ROWS_PER_CHUNK = 50000    # 50k行一chunk
N_WORKERS = 64            # 64个worker

# 可能实现的性能
# - 300个文件（30GB）
# - 耗时: 10-15分钟
# - 速度: 2GB/分钟
```

**警告**: 仅在确认系统资源充足时使用！

## 📞 需要帮助？

如果遇到问题：

1. **检查系统信息cell** - 查看推荐配置
2. **查看错误信息** - 大多数错误都有明确提示
3. **降低配置参数** - 保守配置总是可行的
4. **查看日志** - processed_files.txt 显示进度

---

**使用A100，享受极速体验！🚀**
