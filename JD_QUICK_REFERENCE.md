# JD Processing Quick Reference Card

## 🚀 快速开始（3步）

```bash
# 1. 准备数据
mkdir -p ../JD
mv your_parquet_files/*.parquet ../JD/

# 2. 安装依赖
pip install pandas numpy tqdm skillner pyarrow psutil

# 3. 运行notebook
jupyter notebook extract_skills_from_jd.ipynb
```

## ⚙️ 关键配置参数

| 参数 | 默认值 | 说明 | 何时修改 |
|------|--------|------|----------|
| `BATCH_SIZE` | 10 | 每批处理多少个文件 | 内存不足时减小 |
| `ROWS_PER_CHUNK` | 1000 | 每次处理多少行 | 内存不足时减小 |
| `TEXT_COLUMN` | 'job_description' | 文本列名 | 根据实际列名 |
| `RESUME_FROM_CHECKPOINT` | True | 断点续传 | 重新开始时设False |

## 🧮 内存计算公式

**估算所需内存：**

```
所需内存 ≈ BATCH_SIZE × 单文件大小 × 3
```

**示例：**
- 单个parquet文件：100MB
- BATCH_SIZE = 10
- 所需内存：10 × 100MB × 3 = 3GB

**如果内存不足：**

| 可用内存 | 推荐配置 |
|---------|---------|
| < 4GB | BATCH=1, CHUNK=100 |
| 4-8GB | BATCH=5, CHUNK=500 |
| 8-16GB | BATCH=10, CHUNK=1000 |
| 16GB+ | BATCH=20, CHUNK=5000 |

## 📊 处理流程图

```
输入文件(../JD/*.parquet)
    ↓
[分批处理] BATCH_SIZE个文件一组
    ↓
[每个文件内部分块] 每次ROWS_PER_CHUNK行
    ↓
[提取技能] 使用SkillNER
    ↓
[保存结果] output/extracted_skills/*_processed.parquet
    ↓
[记录检查点] processed_files.txt
    ↓
完成
```

## 🔧 常见问题速查

### 内存溢出

```python
# 方案1: 减小批次
BATCH_SIZE = 5  # 或更小

# 方案2: 减小chunk
ROWS_PER_CHUNK = 500  # 或更小

# 方案3: 只读需要的列
pf = pd.read_parquet(filepath, columns=['job_description', 'job_id'])
```

### 处理被中断

```python
# 自动续传（默认已启用）
RESUME_FROM_CHECKPOINT = True

# 查看进度
processed = len(get_processed_files())
total = len(all_parquet_files)
print(f"{processed}/{total} ({100*processed/total:.1f}%)")
```

### 重新开始

```python
# 方法1: 修改配置
RESUME_FROM_CHECKPOINT = False

# 方法2: 删除检查点
rm output/extracted_skills/processed_files.txt
```

### 处理特定文件

```python
# 从processed_files.txt中移除该文件名
# 然后重新运行notebook
```

## 📈 性能优化速查

| 想要 | 怎么做 |
|------|--------|
| 更快 | 增加 BATCH_SIZE 和 ROWS_PER_CHUNK |
| 更稳定 | 减小 BATCH_SIZE 和 ROWS_PER_CHUNK |
| 省内存 | 只读需要的列 |
| 并行处理 | 使用 joblib.Parallel |
| 处理超大数据 | 使用 Dask |

## 🎯 输出文件说明

**输入：**
```
../JD/jd_part_001.parquet  (原始数据)
```

**输出：**
```
output/extracted_skills/jd_part_001_processed.parquet
```

**新增列：**
- `extracted_skills`: list类型，提取的技能
- `num_skills`: int类型，技能数量

**示例：**
```python
df = pd.read_parquet('output/extracted_skills/jd_part_001_processed.parquet')

# 查看提取结果
df[['job_id', 'extracted_skills', 'num_skills']].head()
```

## 📞 命令速查

```bash
# 查看文件数量
ls ../JD/*.parquet | wc -l

# 查看总大小
du -sh ../JD/

# 查看已处理数量
ls output/extracted_skills/*_processed.parquet | wc -l

# 查看检查点
cat output/extracted_skills/processed_files.txt

# 清理输出（小心使用！）
rm -rf output/extracted_skills/*.parquet

# 重置检查点
rm output/extracted_skills/processed_files.txt
```

## 💾 数据备份建议

```bash
# 处理前备份原始数据
tar -czf JD_backup_$(date +%Y%m%d).tar.gz ../JD/

# 处理后备份结果
tar -czf results_backup_$(date +%Y%m%d).tar.gz output/extracted_skills/

# 只备份处理后的数据（不包括checkpoint）
tar -czf results_$(date +%Y%m%d).tar.gz \
    output/extracted_skills/*_processed.parquet
```

## 🔍 监控命令

```python
# 在notebook中实时监控

# 1. 内存使用
import psutil
process = psutil.Process()
print(f"Memory: {process.memory_info().rss / 1024**3:.2f} GB")

# 2. 处理进度
processed = len(get_processed_files())
total = len(all_parquet_files)
print(f"Progress: {processed}/{total} ({100*processed/total:.1f}%)")

# 3. 估算剩余时间（处理几个文件后）
import time
# 记录开始时间和已处理文件数
# 计算平均处理时间
# 估算剩余时间 = 平均时间 × 剩余文件数
```

## ⚡ 性能基准参考

基于Intel i7处理器，16GB RAM：

| 文件数 | 总大小 | BATCH_SIZE | 总耗时 |
|-------|--------|-----------|--------|
| 50 | 5GB | 10 | ~30分钟 |
| 100 | 10GB | 10 | ~1小时 |
| 200 | 20GB | 5 | ~3小时 |
| 300 | 30GB | 5 | ~5小时 |

**注意：** 实际时间取决于：
- CPU性能
- 硬盘速度（SSD更快）
- 文本长度
- 技能库大小

## 🎓 进阶技巧

### 批量重命名输出文件

```python
# 给所有输出文件添加前缀
from pathlib import Path
for f in Path('output/extracted_skills').glob('*_processed.parquet'):
    new_name = f'skills_{f.name}'
    f.rename(f.parent / new_name)
```

### 合并所有输出（大文件版）

```python
# 使用Dask合并（省内存）
import dask.dataframe as dd

ddf = dd.read_parquet('output/extracted_skills/*_processed.parquet')
ddf.to_parquet('all_skills_combined/', compression='snappy')
```

### 统计分析

```python
# 快速统计（不加载全部数据）
import pandas as pd
from pathlib import Path

stats = []
for f in Path('output/extracted_skills').glob('*_processed.parquet'):
    df = pd.read_parquet(f, columns=['num_skills'])
    stats.append({
        'file': f.name,
        'total_rows': len(df),
        'total_skills': df['num_skills'].sum(),
        'avg_skills': df['num_skills'].mean()
    })

stats_df = pd.DataFrame(stats)
print(stats_df)
print(f"\nGrand Total: {stats_df['total_skills'].sum():,} skills")
```

## 📋 检查清单

处理前：
- [ ] 创建 `../JD/` 目录
- [ ] 移动parquet文件到 `../JD/`
- [ ] 安装所有依赖
- [ ] 检查列名（TEXT_COLUMN, ID_COLUMN）
- [ ] 根据内存调整 BATCH_SIZE

处理中：
- [ ] 监控内存使用
- [ ] 查看控制台输出
- [ ] 检查第一批输出文件
- [ ] 验证提取质量

处理后：
- [ ] 确认所有文件都已处理
- [ ] 检查输出文件数量
- [ ] 抽查几个输出文件
- [ ] 备份结果
- [ ] （可选）删除原始文件

## 🆘 紧急情况

### 处理卡住不动

```python
# 1. 查看是否真的卡住
# 有时只是处理慢，等待几分钟

# 2. 检查CPU使用率
# 如果CPU=100%，说明在正常处理

# 3. 如果确实卡住
# Kernel → Interrupt → Restart
# 然后启用RESUME_FROM_CHECKPOINT继续
```

### 硬盘空间不足

```bash
# 检查空间
df -h

# 清理临时文件
rm -rf ~/.cache/pip
rm -rf /tmp/*

# 压缩已处理文件
tar -czf processed_batch1.tar.gz output/extracted_skills/*.parquet
# 删除原文件释放空间
```

### 结果异常

```python
# 检查是否所有文件都有extracted_skills列
import pandas as pd
from pathlib import Path

for f in Path('output/extracted_skills').glob('*_processed.parquet'):
    df = pd.read_parquet(f)
    if 'extracted_skills' not in df.columns:
        print(f"Missing column in {f.name}")
```

---

**记住：先小规模测试，再大批量处理！**
