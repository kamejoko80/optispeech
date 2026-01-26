import os
import shutil
import argparse


# python3 scripts/prepare_hfc_datasets.py \
#   --train_text "datasets/hi-fi-captain/en-US/female/text/train_parallel.txt" \
#   --train_wav "datasets/hi-fi-captain/en-US/female/wav/train_parallel" \
#   --val_text "datasets/hi-fi-captain/en-US/female/text/eval.txt" \
#   --val_wav "datasets/hi-fi-captain/en-US/female/wav/eval" \
#   --output "datasets/hfc_female-en_us-dataset"


def parse_metadata(text_file):
    """Parses Hi-Fi Captain text files: 'ID Text' -> ['ID', 'Text']"""
    data = []
    if not os.path.exists(text_file):
        print(f"Error: Text file {text_file} not found.")
        return data
    
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                data.append(parts)
    return data

def build_split(name, metadata, src_wav_dir, output_root):
    """Copies wavs and creates metadata.csv for a specific split (train or val)"""
    print(f"Processing {name} split ({len(metadata)} samples)...")
    
    # Create folder structure: output/split/wav
    split_path = os.path.join(output_root, name)
    wav_dest = os.path.join(split_path, "wav")
    os.makedirs(wav_dest, exist_ok=True)
    
    final_metadata = []
    
    for file_id, text in metadata:
        src_wav = os.path.join(src_wav_dir, f"{file_id}.wav")
        dst_wav = os.path.join(wav_dest, f"{file_id}.wav")
        
        if os.path.exists(src_wav):
            shutil.copy2(src_wav, dst_wav)
            # OptiSpeech format: file_id|text
            final_metadata.append(f"{file_id}|{text}")
        else:
            print(f"Warning: {file_id}.wav not found in {src_wav_dir}")

    # Save metadata.csv
    with open(os.path.join(split_path, "metadata.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(final_metadata))

def main():
    parser = argparse.ArgumentParser(description="Prepare OptiSpeech dataset structure.")
    parser.add_argument("--train_text", required=True, help="Path to train_parallel.txt")
    parser.add_argument("--train_wav", required=True, help="Path to train_parallel wav folder")
    parser.add_argument("--val_text", required=True, help="Path to eval.txt")
    parser.add_argument("--val_wav", required=True, help="Path to eval wav folder")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    
    args = parser.parse_args()

    # Process Training Data
    train_meta = parse_metadata(args.train_text)
    build_split("train", train_meta, args.train_wav, args.output)

    # Process Validation Data (using eval folder as val)
    val_meta = parse_metadata(args.val_text)
    build_split("val", val_meta, args.val_wav, args.output)

    print(f"\nSuccess! Dataset built at: {args.output}")

if __name__ == "__main__":
    main()

