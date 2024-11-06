# Key Extractor

A pandas accessor for extracting and formatting keys from DataFrames and Series with support for grouping, batching, and various output formats.

## Features

- Extract keys from DataFrame columns or Series
- Group results by one or multiple columns
- Create batches of fixed size
- Multiple output formats:
  - Series format for further processing
  - String format with customizable separators
  - Direct stdout printing
  - File output
- Support for uniqueness filtering and random sampling

## Usage

### Basic Usage

```python
import pandas as pd
import key_extractor

# DataFrame example
df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B'],
    'value': [1, 2, 3, 4]
})

# Extract values
df.askeys('value', to='str')  # Returns: "1;2;3;4"

# Series example
series = pd.Series([1, 2, 3, 4], name='values')
series.askeys(to='str')  # Returns: "1;2;3;4"
```

### Grouping

```python
# Group by category
result = df.askeys('value', groupby='category', to='str')
# Returns:
# [category: A] (2)
# 1;2
#
# [category: B] (2)
# 3;4
```

### Batching

```python
# Create batches of size 2
result = df.askeys('value', batch_size=2, to='str')
# Returns:
# [batch: 1] (2)
# 1;2
#
# [batch: 2] (2)
# 3;4
```

### Output to File

```python
from pathlib import Path

# Save to files (one per group if grouped)
df.askeys('value', groupby='category', to_file=Path('output'))
```

## Development

### Project Structure

```
key_extractor/

├── key_extractor/
│   ├── __init__.py
│   └── extractor.py
├── tests/
│   └── test_key_extractor.py
└── README.md
```
