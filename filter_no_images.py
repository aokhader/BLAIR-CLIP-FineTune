#!/usr/bin/env python3
"""
Filter Appliances.json to remove data points that have images.
Creates a new file Appliances_No_Images.json with only entries where images list is empty.
"""

import json
import os
from tqdm import tqdm

def filter_no_images(input_file, output_file):
    """
    Filter JSON file to keep only entries with empty images list.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    total_count = 0
    kept_count = 0
    removed_count = 0
    
    # Process file line by line (assuming JSONL format - one JSON object per line)
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        # First pass: count total lines for progress bar
        print("Counting total entries...")
        total_lines = sum(1 for _ in infile)
        infile.seek(0)  # Reset to beginning
        
        print(f"Processing {total_lines:,} entries...")
        
        for line in tqdm(infile, total=total_lines, desc="Filtering"):
            total_count += 1
            
            try:
                # Parse JSON object
                data = json.loads(line.strip())
                
                # Check if images field exists and is empty
                if 'images' in data and len(data['images']) == 0:
                    # Keep this entry - write to output file
                    outfile.write(json.dumps(data) + '\n')
                    kept_count += 1
                else:
                    # Remove this entry (has images)
                    removed_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"\nWarning: Could not parse line {total_count}: {e}")
                continue
    
    # Print summary
    print("\n" + "="*60)
    print("FILTERING SUMMARY")
    print("="*60)
    print(f"Total entries processed:    {total_count:,}")
    print(f"Entries kept (no images):   {kept_count:,} ({kept_count/total_count*100:.2f}%)")
    print(f"Entries removed (w/ images): {removed_count:,} ({removed_count/total_count*100:.2f}%)")
    print("="*60)
    print(f"\nOutput saved to: {output_file}")

def main():
    # Define file paths
    input_file = "AmazonRaw/review_categories/Appliances.json"
    output_file = "AmazonRaw/review_categories/Appliances_No_Images.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print("\nPlease ensure Appliances.json is in the current directory.")
        return
    
    # Get file size
    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    print(f"Input file size: {file_size_mb:.2f} MB")
    
    # Confirm before processing
    print("\nThis will create a new file with only entries that have empty images lists.")
    
    # Run filtering
    filter_no_images(input_file, output_file)

if __name__ == "__main__":
    main()
