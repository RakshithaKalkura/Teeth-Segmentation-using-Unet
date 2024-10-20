import zipfile
import os

# Specify the names of the files to include in the zip
files_to_zip = ['model.py', 'model.pth']

# Create a ZIP file
with zipfile.ZipFile('submission.zip', 'w') as zipf:
    for file in files_to_zip:
        if os.path.isfile(file):
            zipf.write(file)
        else:
            print(f"{file} not found!")

print("submission.zip created successfully!")
