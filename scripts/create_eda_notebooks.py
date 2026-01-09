#!/usr/bin/env python3
"""
Create comprehensive EDA notebooks for PathWild project.
This script generates 6 Jupyter notebooks with complete analysis code.
"""

import json
import os


def create_cell(cell_type, source_lines):
    """Create a notebook cell"""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source_lines if isinstance(source_lines, list) else [source_lines]
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def create_notebook(cells):
    """Create a complete notebook structure"""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def save_notebook(notebook, filepath):
    """Save notebook to file"""
    with open(filepath, 'w') as f:
        json.dump(notebook, f, indent=2)
    print(f"✓ Created {os.path.basename(filepath)}")


# Import all notebook definitions
from notebook_06_cells import get_nb06_cells
from notebook_07_cells import get_nb07_cells
from notebook_08_cells import get_nb08_cells
from notebook_09_cells import get_nb09_cells
from notebook_10_cells import get_nb10_cells
from notebook_11_cells import get_nb11_cells


def main():
    """Generate all notebooks"""
    print("="*70)
    print("PathWild EDA Notebook Generator")
    print("="*70)
    
    notebooks = [
        ("notebooks/06_data_quality_check.ipynb", get_nb06_cells),
        ("notebooks/07_feature_distributions.ipynb", get_nb07_cells),
        ("notebooks/08_spatial_temporal_patterns.ipynb", get_nb08_cells),
        ("notebooks/09_feature_correlations.ipynb", get_nb09_cells),
        ("notebooks/10_heuristic_validation.ipynb", get_nb10_cells),
        ("notebooks/11_target_variable_analysis.ipynb", get_nb11_cells),
    ]
    
    for filepath, get_cells_func in notebooks:
        try:
            cells = get_cells_func()
            notebook = create_notebook(cells)
            save_notebook(notebook, filepath)
        except Exception as e:
            print(f"✗ Error creating {os.path.basename(filepath)}: {e}")
    
    print("="*70)
    print("✓ All notebooks created successfully!")
    print("\nNext steps:")
    print("1. Open notebooks in Jupyter Lab")
    print("2. Run each notebook to generate EDA reports")
    print("3. Review findings before model training")


if __name__ == "__main__":
    main()

