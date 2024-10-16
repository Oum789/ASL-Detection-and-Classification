import zipfile

zip_file_path = 'classification_29classes.zip'
extract_to_path = '.'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)

print(f"Extracted to {extract_to_path}")