# 从句子中提取技能关键词

## 问题场景

ONET 数据中的很多技能都是**完整的句子描述**，例如：

```
"Judgment and Decision Making— Considering the relative costs and benefits of potential actions to choose the most appropriate one."

"Direct or coordinate an organization's financial or budget activities to fund operations, maximize investments, or increase efficiency."

"Accounting software— ComputerEase construction accounting software; Fund accounting software; Intuit QuickBooks; Sage 50 Accounting"
```

如果直接使用这些句子作为技能名称，会导致：
- ❌ 匹配不到简历中的简短表达（如 "problem solving"）
- ❌ 知识库过于冗长，效率低
- ❌ 技能名称不规范

## 解决方案

使用 `extract_skills_from_sentences.py` 脚本，从句子中**智能提取核心技能词汇**。

---

## 🚀 快速使用

### 基本用法（推荐）

```bash
python scripts/extract_skills_from_sentences.py your_onet.json .skillner-kb/ONET_EN.pkl
```

这将使用 `auto` 模式，自动为不同类型的内容选择最佳提取方法。

### 指定提取方法

```bash
# 使用模式匹配（最快，适合格式化的数据）
python scripts/extract_skills_from_sentences.py onet.json output.pkl --method pattern

# 使用 NLP 分析（最准确，需要安装 spaCy）
python scripts/extract_skills_from_sentences.py onet.json output.pkl --method nlp

# 使用关键词提取
python scripts/extract_skills_from_sentences.py onet.json output.pkl --method keyword
```

---

## 📋 提取方法对比

### Method 1: Auto (自动模式) ⭐ 推荐

根据不同的 section 自动选择最佳方法：

| Section | 使用方法 | 说明 |
|---------|---------|------|
| Skills | Pattern | 提取 "—" 前的技能名称 |
| Technology Skills | Pattern + List | 提取类别和具体工具 |
| Work Activities | Pattern | 提取活动名称 |
| Tasks | NLP/Keyword | 提取名词短语 |
| Detailed Work Activities | Pattern | 提取动作-对象对 |

**优点**：
- ✅ 无需配置，开箱即用
- ✅ 针对不同类型数据优化
- ✅ 平衡速度和准确性

**示例输出**：
```python
原句: "Judgment and Decision Making— Considering the relative costs..."
提取: "Judgment and Decision Making"

原句: "Accounting software— QuickBooks; Sage 50; Excel"
提取: ["Accounting software", "QuickBooks", "Sage 50", "Excel"]

原句: "Direct or coordinate an organization's financial activities..."
提取: "Direct financial activities"
```

---

### Method 2: Pattern (模式匹配)

基于文本格式规则提取：
- 提取 "—" 或 "–" 之前的内容
- 对于技术技能，还会按 ";" 分割列表

**优点**：
- ⚡ 最快速
- ✅ 适合格式规范的 ONET 数据
- ✅ 不需要额外依赖

**缺点**：
- ❌ 依赖格式，数据格式不规范时效果差

**适用场景**：
- Skills、Technology Skills、Work Activities 等有明确格式的 section

---

### Method 3: NLP (自然语言处理)

使用 spaCy 进行智能分析：
- 提取名词短语 (noun phrases)
- 识别有意义的技能表达

**安装依赖**：
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

**优点**：
- ✅ 最智能，能理解语义
- ✅ 适合非格式化的描述性文本
- ✅ 能提取多个相关技能

**缺点**：
- ❌ 需要安装额外库（~50MB）
- ❌ 速度较慢

**示例**：
```python
原句: "Direct or coordinate an organization's financial or budget activities to fund operations, maximize investments, or increase efficiency."

提取的名词短语:
- "an organization"
- "financial or budget activities"  ← 相关技能
- "operations"
- "investments"
```

**适用场景**：
- Tasks（任务描述）
- 非格式化的职业描述

---

### Method 4: Keyword (关键词提取)

使用简单启发式规则：
- 提取大写开头的词组
- 识别常见技能模式

**优点**：
- ✅ 无额外依赖
- ✅ 比 pattern 更灵活

**缺点**：
- ❌ 可能提取到非技能词汇
- ❌ 准确性中等

**适用场景**：
- 无法使用 NLP，但数据格式不规范时

---

### Method 5: Action (动作-对象提取)

从任务描述中提取 "动词 + 对象" 对：

**示例**：
```python
原句: "Direct financial operations and manage budget planning."

提取:
- "Direct financial operations"
- "Manage budget planning"
```

**优点**：
- ✅ 适合任务类描述
- ✅ 保留动作语义

**缺点**：
- ❌ 仅适用于任务描述
- ❌ 可能过于具体

---

## 📊 提取效果对比

### 测试数据：ONET Chief Executives

| Section | 原始条目数 | Auto 提取 | Pattern 提取 | NLP 提取 |
|---------|-----------|----------|-------------|---------|
| Skills | 5 | 5 | 5 | 15+ |
| Technology Skills | 3 | 9 | 9 | 12+ |
| Work Activities | 3 | 3 | 3 | 10+ |
| Detailed Work Activities | 3 | 3 | 3 | 8+ |
| Tasks | 2 | 2 | 2 | 6+ |
| **总计** | **16** | **22** | **22** | **50+** |

**分析**：
- **Pattern**: 快速准确，适合格式化数据
- **Auto**: 综合最优，推荐使用
- **NLP**: 提取最多，但可能包含噪音

---

## 🎯 使用建议

### 场景 1: 数据格式规范（有 "—" 分隔符）

**推荐**: `auto` 或 `pattern`

```bash
python scripts/extract_skills_from_sentences.py onet.json output.pkl --method auto
```

**效果**：
- Skills: ✅ 精确提取技能名称
- Technology Skills: ✅ 提取类别 + 具体工具
- Work Activities: ✅ 提取活动名称

---

### 场景 2: 数据包含大量描述性文本

**推荐**: `auto` 或 `nlp`

```bash
# 确保已安装 spaCy
pip install spacy
python -m spacy download en_core_web_sm

# 运行提取
python scripts/extract_skills_from_sentences.py onet.json output.pkl --method auto
```

**效果**：
- Tasks: ✅ 从长句中提取关键短语
- 描述: ✅ 识别有意义的技能表达

---

### 场景 3: 只关心特定 sections

```bash
python scripts/extract_skills_from_sentences.py \
    onet.json output.pkl \
    --sections "Skills,Technology Skills"
```

---

### 场景 4: 需要最大覆盖率

使用 NLP 方法提取尽可能多的技能：

```bash
python scripts/extract_skills_from_sentences.py \
    onet.json output.pkl \
    --method nlp
```

**注意**: 可能需要后处理去除噪音。

---

## 💡 高级用法

### 1. 组合使用两个脚本

**步骤 1**: 先用句子提取脚本

```bash
python scripts/extract_skills_from_sentences.py \
    onet.json \
    .skillner-kb/ONET_SKILLS.pkl \
    --method auto
```

**步骤 2**: 如果需要，再用原始转换脚本补充

```bash
python scripts/convert_onet_to_skillner.py \
    onet.json \
    .skillner-kb/ONET_FULL.pkl \
    --sections "Work Context,Job Zone"
```

**步骤 3**: 合并两个知识库

```python
import pickle

# 加载
with open('.skillner-kb/ONET_SKILLS.pkl', 'rb') as f:
    kb1 = pickle.load(f)
with open('.skillner-kb/ONET_FULL.pkl', 'rb') as f:
    kb2 = pickle.load(f)

# 合并
merged = {}
for kb in [kb1, kb2]:
    for skill, entries in kb.items():
        if skill in merged:
            merged[skill].extend(entries)
        else:
            merged[skill] = entries

# 保存
with open('.skillner-kb/ONET_MERGED.pkl', 'wb') as f:
    pickle.dump(merged, f)
```

---

### 2. 自定义提取规则

编辑 `extract_skills_from_sentences.py`，添加自定义提取函数：

```python
def custom_extract(text: str) -> List[str]:
    """你的自定义提取逻辑"""
    skills = []

    # 示例：提取括号中的内容
    import re
    matches = re.findall(r'\(([^)]+)\)', text)
    skills.extend(matches)

    return skills

# 在 extract_skills_from_onet 函数中使用
if method == 'custom':
    extracted_skills = custom_extract(item)
```

---

### 3. 后处理过滤

提取后可能需要过滤：

```python
import pickle

with open('output.pkl', 'rb') as f:
    kb = pickle.load(f)

# 过滤太短的技能
filtered_kb = {
    skill: entries
    for skill, entries in kb.items()
    if len(skill) >= 3  # 至少3个字符
}

# 过滤常见停用词
stopwords = {'the', 'and', 'or', 'in', 'on', 'at'}
filtered_kb = {
    skill: entries
    for skill, entries in kb.items()
    if skill.lower() not in stopwords
}

# 保存过滤后的结果
with open('output_filtered.pkl', 'wb') as f:
    pickle.dump(filtered_kb, f)
```

---

## 📈 实际应用示例

### 提取前后对比

**原始 ONET 数据（直接使用句子）**：
```python
知识库条目示例:
"judgment and decision making— considering the relative costs and benefits of potential actions to choose the most appropriate one."
```

简历文本："I have strong judgment and decision making skills"

匹配结果：❌ 不匹配（因为简历中没有完整句子）

---

**使用句子提取后**：
```python
知识库条目:
"judgment and decision making"
```

简历文本："I have strong judgment and decision making skills"

匹配结果：✅ 成功匹配！

---

## 🔍 调试和验证

### 查看提取结果

```python
import pickle

with open('.skillner-kb/ONET_EN.pkl', 'rb') as f:
    kb = pickle.load(f)

# 按 section 分组查看
from collections import defaultdict
by_section = defaultdict(list)

for skill_name, entries in kb.items():
    for entry in entries:
        by_section[entry['section']].append(entry['pref_label'])

# 打印每个 section 的技能
for section, skills in by_section.items():
    print(f"\n{section}: {len(skills)} 个技能")
    print("  " + "\n  ".join(sorted(set(skills))[:10]))
```

### 测试提取效果

```bash
# 使用测试数据
python scripts/extract_skills_from_sentences.py \
    tests/test_onet_sample.json \
    tests/output.pkl \
    --method auto

# 查看结果
python -c "
import pickle
kb = pickle.load(open('tests/output.pkl', 'rb'))
for skill in list(kb.keys())[:10]:
    print(f'✓ {kb[skill][0][\"pref_label\"]}')
"
```

---

## ❓ 常见问题

### Q1: 应该使用哪个脚本？

**回答**:
- 如果你的 ONET 数据是**句子描述** → 使用 `extract_skills_from_sentences.py` ⭐
- 如果你的 ONET 数据已经是**简短技能名** → 使用 `convert_onet_to_skillner.py`
- 不确定？→ 使用 `extract_skills_from_sentences.py` 的 `auto` 模式

### Q2: NLP 方法提取了太多技能怎么办？

**解决**:
1. 使用 `auto` 或 `pattern` 方法代替
2. 后处理过滤短词或停用词
3. 只保留出现频率高的技能

### Q3: 如何验证提取效果？

**方法**:
```bash
# 1. 查看统计信息
python scripts/extract_skills_from_sentences.py onet.json output.pkl | grep "Sample"

# 2. 手动检查几个样本
python -c "import pickle; kb = pickle.load(open('output.pkl', 'rb'));
for k, v in list(kb.items())[:5]:
    print(f'{v[0][\"pref_label\"]} <- {v[0][\"original_text\"][:80]}')"

# 3. 实际测试提取效果
python examples/onet_skill_extraction.py
```

### Q4: 可以同时使用多种方法吗？

**可以！** 分别运行不同方法，然后合并知识库：

```bash
python scripts/extract_skills_from_sentences.py onet.json kb1.pkl --method pattern
python scripts/extract_skills_from_sentences.py onet.json kb2.pkl --method nlp

# 然后用 Python 合并
python -c "
import pickle
kb1 = pickle.load(open('kb1.pkl', 'rb'))
kb2 = pickle.load(open('kb2.pkl', 'rb'))
merged = {**kb1, **kb2}
pickle.dump(merged, open('merged.pkl', 'wb'))
"
```

---

## 📚 相关资源

- [ONET Integration Guide](ONET_INTEGRATION.md) - ONET 集成完整文档
- [Quick Start](../ONET_QUICKSTART.md) - 快速开始指南
- [SkillNER Documentation](https://anasaito.github.io/SkillNER) - SkillNER 官方文档

---

## 总结

从句子中提取技能关键词是使用 ONET 数据的**关键步骤**。使用正确的提取方法可以：

✅ 提高技能匹配准确率
✅ 减少知识库冗余
✅ 提升提取性能
✅ 获得更规范的技能名称

**推荐配置**：使用 `auto` 模式，它会自动为不同类型的数据选择最佳方法！

```bash
python scripts/extract_skills_from_sentences.py your_onet.json .skillner-kb/ONET_EN.pkl
```
