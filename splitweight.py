import os

def split_file():
    filepath = "weights/best_model.pth"
    chunk_size = 45 * 1024 * 1024 # 45 MB chunks
    
    print(f"Splitting {filepath}...")
    with open(filepath, 'rb') as f:
        chunk = f.read(chunk_size)
        i = 0
        while chunk:
            part_name = f"{filepath}.part{i}"
            with open(part_name, 'wb') as chunk_file:
                chunk_file.write(chunk)
            print(f"Created {part_name}")
            i += 1
            chunk = f.read(chunk_size)
    print("Done! You can now delete the python script.")

split_file()