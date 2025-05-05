import zipfile, os

# 1) Path to your ZIP file
zip_path   = r"H:\DS&ML Projects\icon-sketch-retrieval\material-design-icons-master.zip"
# 2) Where to dump the 2×48dp PNGs
target_dir = r"H:\DS&ML Projects\icon-sketch-retrieval\data\icons"
os.makedirs(target_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as z:
    for member in z.namelist():
        path = member.replace("\\", "/").lower()
        # match only PNGs in a 48dp/2x folder
        if path.endswith(".png") and "/48dp/2x/" in path:
            filename = os.path.basename(member)
            data = z.read(member)
            with open(os.path.join(target_dir, filename), "wb") as f:
                f.write(data)
            print(f"Extracted: {filename}")

print("Done extracting only the 2×48 dp icons.")