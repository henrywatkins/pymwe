# pymwe

A Python package for finding meaningful multi-word expressions (MWEs) in text using Pointwise Mutual Information.

## Install

```bash
pip install pymwe
```

Or for development:

```bash
# Using rye for package management
git clone https://github.com/username/pymwe.git
cd pymwe
rye sync
```

## Usage

### From the command line

```bash
# Basic usage with default parameters
pymwe-cli textfile.txt

# Custom number of MWEs to find
pymwe-cli --n 5 textfile.txt

# Adjust document frequency thresholds
pymwe-cli --min_df 3 --max_df 0.8 textfile.txt

# Find correlated features for groups/clusters
pymwe-cfeatures groups.txt features.txt --top_k 3 --show_values
```

### From Python

```python
from pymwe.model import find_mwe, cfeatures

# Example with a list of texts
texts = [
    "this is an example of multi word expressions",
    "multi word expressions can be found automatically",
    "meaningful phrases are useful in NLP applications"
]

# Find top 5 multi-word expressions
mwes = find_mwe(texts, n=5, min_df=1, max_df=0.9)
print(mwes)

# Find features correlated with group/cluster IDs
group_ids = [0, 0, 1, 1, 2]
data = [
    ["feature1", "feature2"],
    ["feature1", "feature3"],
    ["feature4", "feature5"],
    ["feature4", "feature6"],
    ["feature7", "feature8"]
]
# Get top 2 features for each group
correlated_features = cfeatures(group_ids, data, top_k=2, show_values=True)
print(correlated_features)
# Output: {0: [('feature1', 1.0), ('feature3', 0.5)], 1: [('feature4', 1.0), ('feature6', 0.5)], 2: [('feature7', 1.0), ('feature8', 1.0)]}
```

## Features

- Fast MWE extraction using PMI (Pointwise Mutual Information)
- Numba-accelerated correlation calculation
- Flexible parameters for document frequency filtering
- Feature correlation analysis using Matthews correlation coefficient
- Simple command-line interface

