import sys
import os
import logging

# Add current directory to sys.path
sys.path.append(os.getcwd())

from baselines.baseline_tfidf import TFIDFBaseline

def run_tfidf():
    print("Running TF-IDF Baseline for Appliances.json (with images)...")
    
    model = TFIDFBaseline(
        data_dir=".", 
        meta_file="meta/meta_Appliances.json",
        reviews_file="AmazonRaw/review_categories/Appliances.json"
    )
    results = model.run()
    print("TF-IDF Results (With Images):", results)

if __name__ == "__main__":
    try:
        run_tfidf()
    except Exception as e:
        print(f"TF-IDF Failed: {e}")
        import traceback
        traceback.print_exc()
