#!/usr/bin/env python3
"""
Convert parquet files with map types to JSONL format.
This script handles parquet files that contain map types which pandas cannot read directly.
Usage: python parquet2jsonl.py /path/to/input/dir -o /path/to/output/dir
"""

import pyarrow.parquet as pq
import pyarrow as pa
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import argparse


def convert_arrow_value_to_python(value: Any) -> Any:
    """
    Recursively convert PyArrow values to Python native types.
    Handles map types, lists, structs, and other complex types.
    """
    if value is None:
        return None
    
    # Handle PyArrow MapArray
    if isinstance(value, pa.MapArray):
        result = {}
        keys = value.keys.to_pylist()
        items = value.items.to_pylist()
        for i in range(len(keys)):
            if keys[i] is not None:
                result[convert_arrow_value_to_python(keys[i])] = convert_arrow_value_to_python(items[i])
        return result
    
    # Handle PyArrow ListArray
    if isinstance(value, pa.ListArray) or isinstance(value, pa.LargeListArray):
        return [convert_arrow_value_to_python(item) for item in value.to_pylist()]
    
    # Handle PyArrow StructArray
    if isinstance(value, pa.StructArray):
        result = {}
        for field in value.type:
            field_value = value.field(field.name)
            if field_value is not None:
                result[field.name] = convert_arrow_value_to_python(field_value)
        return result
    
    # Handle PyArrow Array (scalar)
    if isinstance(value, pa.Array):
        if len(value) == 0:
            return None
        if len(value) == 1:
            return convert_arrow_value_to_python(value[0].as_py())
        return [convert_arrow_value_to_python(item.as_py()) for item in value]
    
    # Handle PyArrow Scalar
    if isinstance(value, pa.Scalar):
        return convert_arrow_value_to_python(value.as_py())
    
    # Handle nested dicts and lists
    if isinstance(value, dict):
        return {k: convert_arrow_value_to_python(v) for k, v in value.items()}
    
    if isinstance(value, list):
        return [convert_arrow_value_to_python(item) for item in value]
    
    # Return primitive types as-is
    return value


def convert_parquet_to_jsonl(parquet_file: Path, jsonl_file: Path):
    """
    Convert a parquet file to JSONL format, handling map types.
    """
    print(f"Converting {parquet_file} -> {jsonl_file}")
    
    # Read parquet file using PyArrow
    parquet_file_obj = pq.ParquetFile(parquet_file)
    table = parquet_file_obj.read()
    
    # Create output directory if needed
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get column names
    column_names = table.column_names
    
    # Process each row
    num_rows = table.num_rows
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for i in range(num_rows):
            row_dict = {}
            for col_name in column_names:
                col = table[col_name]
                value = col[i]
                
                # Convert PyArrow value to Python native type
                python_value = convert_arrow_value_to_python(value)
                row_dict[col_name] = python_value
            
            # Write as JSON line
            json_line = json.dumps(row_dict, ensure_ascii=False, default=str)
            f.write(json_line + '\n')
            
            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{num_rows} rows...", end='\r')
    
    print(f"\n  Completed: {num_rows} rows converted")
    return num_rows


def convert_all_parquet_files(input_dir: Path, output_dir: Path = None):
    """
    Convert all parquet files in a directory to JSONL format.
    """
    if output_dir is None:
        output_dir = input_dir / "jsonl_output"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all parquet files
    parquet_files = sorted(input_dir.glob("*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet file(s) to convert.")
    
    total_rows = 0
    for parquet_file in parquet_files:
        jsonl_file = output_dir / f"{parquet_file.stem}.jsonl"
        try:
            rows = convert_parquet_to_jsonl(parquet_file, jsonl_file)
            total_rows += rows
        except Exception as e:
            print(f"\nError converting {parquet_file}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nTotal: {total_rows} rows converted from {len(parquet_files)} files")


def main():
    parser = argparse.ArgumentParser(
        description="Convert parquet files with map types to JSONL format"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing parquet files"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for JSONL files (default: input_dir/jsonl_output)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    output_dir = Path(args.output) if args.output else None
    convert_all_parquet_files(input_dir, output_dir)


if __name__ == "__main__":
    main()

