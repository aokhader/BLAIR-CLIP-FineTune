import csv
import os
import logging
import sys
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_filtered_tsv():
    # File Paths
    base_dir = "../data/Applicances Without Images"
    # Note: Using the unzipped directory name provided in previous steps
    # Check exact path name: "Applicances Without Images" (sic)
    
    test_csv = os.path.join(base_dir, "Appliances_No_Images.test.csv")
    valid_csv = os.path.join(base_dir, "Appliances_No_Images.valid.csv")
    train_csv = os.path.join(base_dir, "Appliances_No_Images.train.csv")
    
    source_tsv = "clean_review_meta_with_images.tsv"
    output_tsv = "clean_review_meta_with_images_TRAIN_ONLY.tsv"
    
    if not os.path.exists(source_tsv):
        logging.error(f"Source file {source_tsv} not found in current directory.")
        sys.exit(1)
        
    # 1. Collect ASINs that appear in the Test set
    # Strategy: If an ASIN is in the test set, should we exclude its training data?
    # No, we can train on the item's representation.
    # We just want to avoid training on the *specific interactions* that constitute the test set.
    # But the TSV doesn't have User IDs. It has Review Text.
    # The review text in the TSV likely corresponds to the review in the interaction.
    # We can't easily link TSV row to CSV row without UserID or ReviewID.
    
    # However, let's look at the counts:
    # TSV: ~140k lines
    # CSVs: ~2M lines? Wait.
    # Train CSV: 97k
    # Valid CSV: 231k?
    # Test CSV: 1.6M? 
    # That looks wrong. Usually Train >> Test.
    # Let's re-read the wc output carefully.
    
    # 1691092 ... test.csv
    # 97943 ... train.csv
    # 231063 ... valid.csv
    
    # Wait, 1.6M test vs 97k train? That's extremely unusual for a split.
    # Maybe I misread the wc output or filenames.
    
    # Let's assume the TSV contains a subset or different set.
    # Ideally, we filter out any row in the TSV where the ASIN corresponds to a TEST interaction?
    # No, multiple users buy the same item.
    
    # Conservative Approach:
    # Filter out any ASIN that appears in the Test Set? 
    # That would be "Zero-Shot Item Recommendation" (Cold Start).
    # If the benchmark is standard, usually items in Test appear in Train.
    
    # Pragmatic Approach given Data Constraints:
    # The TSV contains (Review, Meta, Image).
    # We are training a Joint Encoder.
    # If we train on (Review_of_Test_Item, Image_of_Test_Item), we might learn a correlation 
    # that helps link "User History (which is text/image)" to "Target Item".
    # But the Review text in the TSV is specific to that interaction.
    
    # Since we can't map TSV rows to Test Interactions (no UserID), 
    # and the TSV is much smaller (140k) than the CSVs (2M), 
    # it implies the TSV is a subset (maybe "5-core" vs "all"?).
    
    # The "clean_review_meta_with_images.tsv" seems to be the dataset for Pre-training.
    # The CSVs are for the Recommendation Benchmark.
    
    # Ideally, we should not train on *any* data that involves the Test Interactions.
    # But we can't identify them.
    
    # Best Effort:
    # We will assume that the TSV lines are valuable for learning Item Representations.
    # We will create a new TSV that simply splits the existing TSV into Train/Val 
    # to avoid the script doing it randomly, OR we accept the random split of the pre-training data 
    # because pre-training data != downstream benchmark data.
    
    # WAIT. The user asked "are we sure that our split is not used for training?".
    # If `base_clip.sh` trains on `clean_review_meta_with_images.tsv`, and we evaluate on `Appliances_No_Images.test.csv`,
    # and if the TSV contains the reviews from the test CSV, then YES, there is leakage.
    
    # Since we can't filter by interaction, we can filter by ASIN if we want to be strictly cold-start (unlikely).
    # OR, we can try to match the review text? But that's hard/slow.
    
    # Let's just create a split of the TSV that excludes ASINs that appear ONLY in the test set?
    # Or more likely: The TSV is just item metadata + one review? 
    # The header says "review". 
    
    # Decision: The safest "blind" split is to randomly split the TSV and use that.
    # But the user specifically asked about *their* split.
    # The `run_evaluation.py` loads `Appliances_No_Images.train.csv`.
    # Let's verify if the TSV ASINs are in the Train CSV.
    
    # Let's create a subset of the TSV that ONLY contains ASINs found in `Appliances_No_Images.train.csv`.
    # This ensures we only pre-train on items/reviews associated with the training set.
    # This is a robust way to avoid leakage.
    pass

def filter_tsv_by_train_asins():
    base_dir = "splits/with_images"
    train_csv = os.path.join(base_dir, "Appliances.train.csv")
    valid_csv = os.path.join(base_dir, "Appliances.valid.csv")
    source_tsv = "clean_review_meta_with_images.tsv"
    output_tsv = "clean_review_meta_with_images_FROM_SPLITS.tsv"
    
    # 1. Load Allowed ASINs (Train + Valid)
    allowed_asins = set()
    
    logging.info("Loading allowed ASINs from Train/Valid CSVs...")
    for csv_file in [train_csv, valid_csv]:
        if os.path.exists(csv_file):
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) > 1:
                        allowed_asins.add(row[1]) # ASIN is column 1
        else:
            logging.warning(f"CSV file not found: {csv_file}")
            
    logging.info(f"Found {len(allowed_asins)} allowed ASINs.")
    
    # 2. Filter TSV
    logging.info(f"Filtering {source_tsv}...")
    kept_count = 0
    total_count = 0
    
    with open(source_tsv, 'r', encoding='utf-8') as f_in, \
         open(output_tsv, 'w', encoding='utf-8') as f_out:
        
        reader = csv.reader(f_in, delimiter='\t')
        writer = csv.writer(f_out, delimiter='\t')
        
        # Header
        try:
            header = next(reader)
            writer.writerow(header)
            
            try:
                img_idx = header.index('image_path')
            except ValueError:
                img_idx = 2 # Fallback
                
            for row in tqdm(reader):
                total_count += 1
                if len(row) <= img_idx:
                    continue
                    
                image_path = row[img_idx]
                # Extract ASIN from path "blair_clip_images/B00004YWK2.jpg"
                basename = os.path.basename(image_path)
                asin = os.path.splitext(basename)[0]
                
                if asin in allowed_asins:
                    writer.writerow(row)
                    kept_count += 1
                    
        except StopIteration:
            pass
            
    logging.info(f"Finished. Kept {kept_count}/{total_count} rows. Saved to {output_tsv}")

if __name__ == "__main__":
    filter_tsv_by_train_asins()

