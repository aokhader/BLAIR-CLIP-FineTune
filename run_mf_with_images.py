import sys
import os
import logging

# Add current directory to sys.path
sys.path.append(os.getcwd())

from mf_baseline.baseline_mf import MFBaseline

def run_mf():
    print("Running MF Baseline for Appliances.json (with images)...")
    
    # MFBaseline defaults to epochs=10.
    # We point it to the original Appliances.json file.
    model = MFBaseline(
        data_dir=".",
        meta_file="meta/meta_Appliances.json",
        reviews_file="AmazonRaw/review_categories/Appliances.json",
        epochs=10
    )
    results = model.run()
    print("MF Results (With Images):", results)

if __name__ == "__main__":
    try:
        run_mf()
    except Exception as e:
        print(f"MF Failed: {e}")
        import traceback
        traceback.print_exc()
