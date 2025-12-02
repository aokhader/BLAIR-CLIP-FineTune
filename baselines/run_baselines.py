import sys
import os
import logging

# Add current directory to sys.path
sys.path.append(os.getcwd())

from baseline_tfidf import TFIDFBaseline
from baseline_mf import MFBaseline

def run_tfidf(data_dir: str, meta_file: str, reviews_file: str):
    model = TFIDFBaseline(
        data_dir=data_dir, 
        meta_file=meta_file,
        reviews_file=reviews_file
    )
    results = model.run()
    print("TF-IDF Results:", results)

def run_mf(data_dir: str, meta_file: str, reviews_file: str):
    model = MFBaseline(
        data_dir=data_dir,
        meta_file=meta_file,
        reviews_file=reviews_file,
        epochs=10
    )
    results = model.run()
    print("MF Results:", results)

if __name__ == "__main__":
    try:
        print("Running TF-IDF Baseline for Appliances WITHOUT Images...")
        run_tfidf(".", "meta/meta_Appliances.json", "AmazonRaw/review_categories/Appliances_No_Images.json")
    except Exception as e:
        print(f"TF-IDF Failed: {e}")
        import traceback
        traceback.print_exc()

    print("-" * 50)
    
    try:
        print("Running MF Baseline for Appliances WITHOUT Images...")
        run_mf(".", "meta/meta_Appliances.json", "AmazonRaw/review_categories/Appliances_No_Images.json")
    except Exception as e:
        print(f"MF Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("-" * 50)

    try:
        print("Running TF-IDF Baseline for Appliances WITH Images...")
        run_tfidf(".", "meta/meta_Appliances.json", "AmazonRaw/review_categories/Appliances.json")
    except Exception as e:
        print(f"TF-IDF Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("-" * 50)
    
    try:
        print("Running MF Baseline for Appliances WITH Images...")
        run_mf(".", "meta/meta_Appliances.json", "AmazonRaw/review_categories/Appliances.json")
    except Exception as e:
        print(f"MF Failed: {e}")
        import traceback
        traceback.print_exc()
    