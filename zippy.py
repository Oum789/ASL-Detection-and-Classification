import zipfile
import os

# Path to your zip file and extract destination
zip_file_path = 'classification_29classes.zip'
extract_to_path = '.'

# Create a ZipFile object and extract its contents
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)

print(f"Extracted to {extract_to_path}")