import os

def rename_audio_files(parent_folder):
    # Walk through all subdirectories
    for root, dirs, files in os.walk(parent_folder):
        for filename in files:
            # Process only WAV files
            if filename.endswith(".wav"):
                old_path = os.path.join(root, filename)

                # Example: "file3.mp3.wav_16k.wav_norm.wav_mono.wav_silence.wav"
                # Split into name + final extension
                parts = filename.split(".")
                
                # If the file has no extra dots, skip it
                if len(parts) < 2:
                    continue

                # Keep last extension only (wav)
                *name_parts, ext = parts

                # Join everything in name_parts with underscores
                new_name = "_".join(name_parts) + "." + ext

                new_path = os.path.join(root, new_name)

                # Rename only if different
                if old_path != new_path:
                    print(f"Renaming:\n  {old_path}\n  -> {new_path}\n")
                    os.rename(old_path, new_path)


if __name__ == "__main__":
    folder = "/Users/shanoonissaka/Documents/school/thesis-project/datasets/audio/for-norm/validation"
    rename_audio_files(folder)
