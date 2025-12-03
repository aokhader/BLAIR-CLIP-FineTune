import sys
import os
import logging

# Add current directory to sys.path
sys.path.append(os.getcwd())

from baselines.baseline_tfidf import TFIDFBaseline
from mf_baseline.baseline_mf import MFBaseline

def run_tfidf():
    print("Running TF-IDF Baseline for Appliances_no_images...")
    model = TFIDFBaseline(
        data_dir=".", 
        meta_file="meta/meta_Appliances.json",
        reviews_file="AmazonRaw/review_categories/Appliances_no_images.json"
    )
    results = model.run()
    print("TF-IDF Results:", results)

def run_mf():
    print("Running MF Baseline for Appliances_no_images...")
    model = MFBaseline(
        data_dir=".",
        meta_file="meta/meta_Appliances.json",
        reviews_file="AmazonRaw/review_categories/Appliances_no_images.json",
        epochs=10
    )
    results = model.run()
    print("MF Results:", results)

if __name__ == "__main__":
    try:
        run_tfidf()
    except Exception as e:
        print(f"TF-IDF Failed: {e}")
        import traceback
        traceback.print_exc()

    print("-" * 50)
    
    try:
        run_mf()
    except Exception as e:
        print(f"MF Failed: {e}")
        import traceback
        traceback.print_exc()
