# ONET 知识库快速开始指南

## 🎯 一分钟上手

### ⚠️ 重要：选择正确的转换脚本

ONET 数据中的技能通常是**完整句子**，如：
```
"Judgment and Decision Making— Considering the relative costs..."
"Direct or coordinate an organization's financial activities..."
```

**推荐使用** `extract_skills_from_sentences.py` 来提取核心技能词汇！

### 1️⃣ 方法A：从句子中提取技能（推荐）⭐

```bash
python scripts/extract_skills_from_sentences.py your_onet_data.json .skillner-kb/ONET_EN.pkl
```

这会从句子中智能提取核心技能关键词，如：
- 原句："Judgment and Decision Making— Considering..." → 提取："Judgment and Decision Making"
- 原句："Accounting software— QuickBooks; Sage 50" → 提取："QuickBooks", "Sage 50"

### 1️⃣ 方法B：直接转换（如果技能已经很简短）

```bash
python scripts/convert_onet_to_skillner.py your_onet_data.json .skillner-kb/ONET_EN.pkl
```

### 2️⃣ 使用 ONET 进行技能提取

```python
import pickle
from skillner.core import Pipeline, Document
from skillner.text_loaders import StrTextLoader
from skillner.matchers import SlidingWindowMatcher
from skillner.conflict_resolvers import SpanProcessor

# 加载知识库
with open('.skillner-kb/ONET_EN.pkl', 'rb') as f:
    onet_kb = pickle.load(f)

# 创建查询函数
def query_method(query: str):
    return onet_kb.get(query.lower(), [])

# 待提取的文本
text = "I have experience with Microsoft Excel, complex problem solving, and critical thinking."

# 构建提取管道
doc = Document()
pipeline = Pipeline()

pipeline.add_node(StrTextLoader(text), name='loader')
pipeline.add_node(
    SlidingWindowMatcher(query_method, max_window_size=5,
                        pre_filter=lambda w: w.lower()),
    name='matcher'
)
pipeline.add_node(
    SpanProcessor(dict_filters={'max_candidate': lambda span: max(span.li_candidates, key=len)}),
    name='resolver'
)

# 运行提取
pipeline.run(doc)

# 显示结果
for sentence in doc:
    for span in sentence.li_spans:
        candidate = span.metadata.get('max_candidate')
        if candidate:
            skill = ' '.join(sentence[candidate.window])
            print(f"✓ {skill}")
```

## 📝 ONET JSON 格式要求

你的 ONET JSON 应该是这样的结构（单个职业或职业数组）：

```json
[
  {
    "code": "11-1011.00",
    "title": "Chief Executives",
    "sections": {
      "Skills": {
        "type": "list",
        "items": ["Judgment and Decision Making— ...", ...]
      },
      "Technology Skills": {
        "type": "list",
        "items": ["Accounting software— QuickBooks; ...", ...]
      }
    }
  }
]
```

## 🔧 自定义选项

### 只提取特定 sections

```bash
python scripts/convert_onet_to_skillner.py \
    your_onet.json \
    output.pkl \
    --sections "Skills,Technology Skills"
```

### 合并 ESCO 和 ONET 知识库

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
            merged_kb[skill_name] = entries

# 使用合并后的知识库进行提取
def query_method(query: str):
    return merged_kb.get(query.lower(), [])
```

## 📚 更多示例

运行完整示例：

```bash
python examples/onet_skill_extraction.py
```

查看详细文档：

```bash
cat docs/ONET_INTEGRATION.md
```

## 🧪 测试

运行测试验证集成：

```bash
# 转换测试数据
python scripts/convert_onet_to_skillner.py \
    tests/test_onet_sample.json \
    tests/test_onet_output.pkl

# 验证输出
python -c "import pickle; kb = pickle.load(open('tests/test_onet_output.pkl', 'rb')); print(f'Loaded {len(kb)} skills')"
```

## 📊 转换后的知识库格式

转换后的知识库是一个字典：

```python
{
    "microsoft excel": [  # 技能名称（小写，用于匹配）
        {
            "concept_id": "onet:11-1011.00:Technology Skills:15:Microsoft Excel",
            "pref_label": "Microsoft Excel",  # 原始大小写
            "occupation_code": "11-1011.00",
            "occupation_title": "Chief Executives",
            "section": "Technology Skills",
            "original_text": "Spreadsheet software— Microsoft Excel"
        }
    ]
}
```

## ❓ 常见问题

**Q: 我的 ONET JSON 格式不一样怎么办？**

A: 修改 `scripts/convert_onet_to_skillner.py` 中的解析函数来适配你的格式。

**Q: 如何提高匹配准确性？**

A: 增加 `max_window_size` 参数（默认5，可以调到6-8）来匹配更长的技能名称。

**Q: 可以和 ESCO 一起使用吗？**

A: 可以！见上面的"合并知识库"示例。

## 🎉 完成！

现在你可以使用 ONET 知识库进行技能提取了！如有问题，请查看 `docs/ONET_INTEGRATION.md` 获取详细文档。
