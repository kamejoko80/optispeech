import os
import shutil
import random

# python3 scripts/split_datasets.py 

# --- CONFIGURATION ---
SOURCE_TEXT = "datasets/hi-fi-captain/en-US/male/text/train_parallel.txt"
SOURCE_WAV_DIR = "datasets/hi-fi-captain/en-US/male/wav/train_parallel"
DEST_ROOT = "datasets/hfc_male-en_us-dataset"
VAL_RATIO = 0.05  # 5% for validation

def prepare_optispeech_data():
    # 1. Read the original text file
    with open(SOURCE_TEXT, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Clean and parse lines (Handling space-separated ID and Text)
    data = []
    for line in lines:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            file_id, text = parts
            data.append((file_id, text))

    # 2. Shuffle and Split
    random.shuffle(data)
    val_size = int(len(data) * VAL_RATIO)
    val_data = data[:val_size]
    train_data = data[val_size:]

    # 3. Define folder creation helper
    def process_split(split_name, split_data):
        split_path = os.path.join(DEST_ROOT, split_name)
        wav_dest_path = os.path.join(split_path, "wav")
        
        os.makedirs(wav_dest_path, exist_ok=True)
        
        metadata_rows = []
        for file_id, text in split_data:
            # OptiSpeech expects file_id|text
            metadata_rows.append(f"{file_id}|{text}")
            
            # Copy wav file
            src_wav = os.path.join(SOURCE_WAV_DIR, f"{file_id}.wav")
            dst_wav = os.path.join(wav_dest_path, f"{file_id}.wav")
            
            if os.path.exists(src_wav):
                shutil.copy2(src_wav, dst_wav)
            else:
                print(f"Warning: File {src_wav} not found!")

        # Write metadata.csv
        with open(os.path.join(split_path, "metadata.csv"), "w", encoding="utf-8") as f:
            f.write("\n".join(metadata_rows))

    # 4. Execute for both splits
    print(f"Processing {len(train_data)} training samples...")
    process_split("train", train_data)
    
    print(f"Processing {len(val_data)} validation samples...")
    process_split("val", val_data)

    print("\nDone! Your dataset is ready in:", DEST_ROOT)

if __name__ == "__main__":
    prepare_optispeech_data()
