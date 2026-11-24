import os

# نام فایل خروجی که تمام کدها در آن جمع می‌شود
output_file = "all_neuropolis_code.txt"

# فرمت‌هایی که می‌خواهید به هوش مصنوعی بدهید
allowed_exts = [".py", ".yaml", ".md"]

# پوشه‌هایی که باید نادیده گرفته شوند
skip_dirs = [".git", "__pycache__", "models", "results", "artifacts", ".venv"]

with open(output_file, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk("."):
        # حذف پوشه‌های اضافی از پیمایش
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if any(file.endswith(ext) for ext in allowed_exts) and file != "merge_code.py":
                filepath = os.path.join(root, file)
                
                # ایجاد یک جداکننده تمیز با نام فایل برای هوش مصنوعی
                outfile.write(f"\n\n{'='*80}\n")
                outfile.write(f"FILE_PATH: {filepath}\n")
                outfile.write(f"{'='*80}\n\n")
                
                try:
                    with open(filepath, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"[ERROR READING FILE: {e}]\n")

print(f"Done! All code merged into '{output_file}'")