# ONET Knowledge Base Integration Guide

本指南介绍如何将 O*NET (Occupational Information Network) 技能库集成到 SkillNER 中进行技能提取。

## 目录
- [概述](#概述)
- [快速开始](#快速开始)
- [详细步骤](#详细步骤)
- [使用示例](#使用示例)
- [高级用法](#高级用法)
- [常见问题](#常见问题)

## 概述

O*NET 是美国劳工部开发的职业信息数据库，包含了详细的职业技能、工作活动、技术要求等信息。通过将 O*NET 数据集成到 SkillNER，你可以：

- 使用美国标准的职业技能分类进行技能提取
- 提取技术技能（如软件、工具等）
- 识别工作活动和详细工作任务
- 结合 ESCO 和 O*NET 实现更全面的技能识别

## 快速开始

### 1. 准备 O*NET JSON 数据

确保你的 O*NET 数据是 JSON 格式，结构如下：

```json
[
  {
    "code": "11-1011.00",
    "title": "Chief Executives",
    "url": "https://www.onetonline.org/link/summary/11-1011.00",
    "sections": {
      "Skills": {
        "type": "list",
        "items": [
          "Judgment and Decision Making— Considering...",
          "Complex Problem Solving— Identifying..."
        ]
      },
      "Technology Skills": {
        "type": "list",
        "items": [
          "Accounting software— QuickBooks; Sage 50",
          "Spreadsheet software— Microsoft Excel"
        ]
      }
    }
  }
]
```

### 2. 转换为 SkillNER 格式

运行转换脚本：

```bash
python scripts/convert_onet_to_skillner.py your_onet_data.json .skillner-kb/ONET_EN.pkl
```

### 3. 使用 O*NET 进行技能提取

```python
import pickle
from skillner.core import Pipeline, Document
from skillner.text_loaders import StrTextLoader
from skillner.matchers import SlidingWindowMatcher
from skillner.conflict_resolvers import SpanProcessor

# 加载知识库
with open('.skillner-kb/ONET_EN.pkl', 'rb') as f:
    onet_kb = pickle.load(f)

# 创建查询方法
def query_method(query: str):
    return onet_kb.get(query.lower(), [])

# 构建提取管道
text = "I have experience with Microsoft Excel and complex problem solving."
doc = Document()

pipeline = Pipeline()
pipeline.add_node(StrTextLoader(text), name='loader')
pipeline.add_node(
    SlidingWindowMatcher(query_method, max_window_size=5,
                        pre_filter=lambda w: w.lower()),
    name='matcher'
)
pipeline.add_node(
    SpanProcessor(dict_filters={"max_candidate": lambda span: max(span.li_candidates, key=len)}),
    name="resolver"
)

pipeline.run(doc)

# 显示结果
for sentence in doc:
    for span in sentence.li_spans:
        candidate = span.metadata.get('max_candidate')
        if candidate:
            skill = " ".join(sentence[candidate.window])
            print(f"Found skill: {skill}")
```

## 详细步骤

### 步骤 1: 数据准备

O*NET 数据可以从以下来源获取：
- O*NET OnLine 网站爬虫
- O*NET 官方数据库导出
- 第三方 API

确保你的 JSON 文件包含以下关键字段：

| 字段 | 说明 | 必需 |
|------|------|------|
| `code` | 职业代码 (如 "11-1011.00") | 是 |
| `title` | 职业名称 | 是 |
| `sections` | 包含技能数据的各个部分 | 是 |

### 步骤 2: 选择要提取的部分

转换脚本支持从以下 sections 提取技能：

| Section | 内容 | 建议使用 |
|---------|------|----------|
| `Skills` | 核心技能（如判断力、问题解决） | ✅ 推荐 |
| `Technology Skills` | 技术技能（软件、工具） | ✅ 推荐 |
| `Work Activities` | 工作活动 | ✅ 推荐 |
| `Detailed Work Activities` | 详细工作活动 | ⚠️ 可选 |
| `Tasks` | 具体任务 | ⚠️ 可选（可能太具体）|

默认提取：`Skills`, `Technology Skills`, `Work Activities`, `Detailed Work Activities`

自定义提取的 sections：

```bash
python scripts/convert_onet_to_skillner.py \
    your_onet.json \
    .skillner-kb/ONET_EN.pkl \
    --sections "Skills,Technology Skills"
```

### 步骤 3: 运行转换

基本用法：

```bash
# 单个职业文件
python scripts/convert_onet_to_skillner.py onet_single.json output.pkl

# 多个职业文件（JSON 数组）
python scripts/convert_onet_to_skillner.py onet_multiple.json output.pkl

# 指定特定 sections
python scripts/convert_onet_to_skillner.py \
    onet_data.json \
    .skillner-kb/ONET_EN.pkl \
    --sections "Skills,Technology Skills,Work Activities"
```

转换完成后，你会看到统计信息：

```
Processing 150 occupation(s)...
Extracting from sections: Skills, Technology Skills, Work Activities

  - Chief Executives: 245 unique skills
  - Software Developers: 189 unique skills
  ...

Total unique skills: 8,432

=== Statistics ===
Total unique skill terms: 8,432
Total skill entries: 45,678
```

### 步骤 4: 验证转换结果

检查生成的知识库：

```python
import pickle

with open('.skillner-kb/ONET_EN.pkl', 'rb') as f:
    kb = pickle.load(f)

print(f"Total skills: {len(kb)}")

# 查看示例
sample_skill = kb['microsoft excel'][0]
print(sample_skill)
# {
#   'concept_id': 'onet:11-1011.00:Technology Skills:15:Microsoft Excel',
#   'pref_label': 'Microsoft Excel',
#   'occupation_code': '11-1011.00',
#   'occupation_title': 'Chief Executives',
#   'section': 'Technology Skills',
#   'original_text': 'Spreadsheet software— Microsoft Excel'
# }
```

## 使用示例

### 示例 1: 基本技能提取

参见 `examples/onet_skill_extraction.py`:

```bash
python examples/onet_skill_extraction.py
```

### 示例 2: 合并 ESCO 和 O*NET

```python
import pickle

# 加载两个知识库
with open('.skillner-kb/ESCO_EN.pkl', 'rb') as f:
    esco_kb = pickle.load(f)

with open('.skillner-kb/ONET_EN.pkl', 'rb') as f:
    onet_kb = pickle.load(f)

# 合并
merged_kb = {}
for kb in [esco_kb, onet_kb]:
    for skill_name, entries in kb.items():
        if skill_name in merged_kb:
            merged_kb[skill_name].extend(entries)
        else:
            merged_kb[skill_name] = entries.copy()

print(f"ESCO: {len(esco_kb)} skills")
print(f"O*NET: {len(onet_kb)} skills")
print(f"Merged: {len(merged_kb)} skills")

# 保存合并后的知识库
with open('.skillner-kb/MERGED_EN.pkl', 'wb') as f:
    pickle.dump(merged_kb, f)
```

### 示例 3: 从简历中提取技能

```python
resume_text = """
Senior Software Engineer with 8 years of experience in Python, Java, and JavaScript.
Expert in complex problem solving, critical thinking, and judgment and decision making.
Proficient with Microsoft Excel, Microsoft PowerPoint, and database management systems.
Strong coordination and negotiation skills with proven track record in team leadership.
"""

# 使用 O*NET 知识库提取技能
# (见上面的快速开始示例)
```

## 高级用法

### 自定义技能解析

如果你的 O*NET 数据格式不同，可以修改 `convert_onet_to_skillner.py` 中的解析函数：

```python
def custom_parse_skill(skill_text: str) -> List[str]:
    """
    自定义技能解析逻辑
    """
    # 你的解析代码
    return skills

# 在 extract_skills_from_onet 函数中使用
if section_name == 'Your Custom Section':
    skill_names = custom_parse_skill(item)
```

### 添加额外的元数据

在转换时可以添加更多元数据：

```python
skill_entry = {
    'concept_id': f'onet:{occupation_code}:{section_name}:{idx}:{skill_name}',
    'pref_label': skill_name,
    'occupation_code': occupation_code,
    'occupation_title': occupation_title,
    'section': section_name,
    'original_text': item,
    # 添加自定义字段
    'importance_level': extract_importance(item),  # 你的自定义函数
    'skill_category': categorize_skill(skill_name),
}
```

### 技能标准化和去重

```python
def normalize_skill_name(skill: str) -> str:
    """
    标准化技能名称，提高匹配准确性
    """
    # 转小写
    skill = skill.lower()

    # 移除版本号: "Python 3.9" -> "Python"
    skill = re.sub(r'\s+\d+(\.\d+)*\s*$', '', skill)

    # 移除公司名前缀: "Microsoft Excel" -> "Excel" (可选)
    # skill = re.sub(r'^(microsoft|adobe|google|oracle)\s+', '', skill)

    return skill.strip()
```

## 常见问题

### Q1: 转换后的知识库太大怎么办？

**A**: 你可以：
1. 只选择特定的 sections（如只要 Skills 和 Technology Skills）
2. 过滤掉不常用的技能
3. 只转换特定职业类别的数据

```bash
# 只提取核心技能
python scripts/convert_onet_to_skillner.py \
    onet_data.json \
    .skillner-kb/ONET_CORE.pkl \
    --sections "Skills,Technology Skills"
```

### Q2: 如何提高匹配准确性？

**A**:
1. 调整 `max_window_size` 参数（技能名称越长，值越大）
2. 使用更好的 `pre_filter` 函数
3. 添加同义词和变体到知识库

```python
# 调整窗口大小
SlidingWindowMatcher(
    query_method,
    max_window_size=6,  # 增加以匹配更长的技能名
    pre_filter=lambda w: w.lower()
)
```

### Q3: 技能重复怎么办？

**A**: SkillNER 的设计允许一个技能对应多个条目（不同职业），这是正常的。你可以在后处理中去重：

```python
def deduplicate_skills(extracted_skills):
    """
    根据 concept_id 或 pref_label 去重
    """
    seen = set()
    unique_skills = []

    for skill in extracted_skills:
        key = skill['pref_label'].lower()
        if key not in seen:
            seen.add(key)
            unique_skills.append(skill)

    return unique_skills
```

### Q4: 如何更新知识库？

**A**: 重新运行转换脚本即可：

```bash
# 下载最新的 O*NET 数据
# 然后重新转换
python scripts/convert_onet_to_skillner.py new_onet_data.json .skillner-kb/ONET_EN.pkl
```

### Q5: 可以同时使用多个知识库吗？

**A**: 可以！有两种方式：

**方式 1**: 合并为一个知识库（推荐）
```python
merged_kb = merge_knowledge_bases([esco_kb, onet_kb, custom_kb])
```

**方式 2**: 使用多个 matcher
```python
pipeline.add_node(SlidingWindowMatcher(esco_query_method), name='esco_matcher')
pipeline.add_node(SlidingWindowMatcher(onet_query_method), name='onet_matcher')
```

## 相关资源

- [SkillNER 文档](https://anasaito.github.io/SkillNER)
- [O*NET OnLine](https://www.onetonline.org/)
- [ESCO 技能分类](https://ec.europa.eu/esco/portal)

## 贡献

如果你发现问题或有改进建议，欢迎提交 Issue 或 Pull Request！
