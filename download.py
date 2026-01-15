import os
import shutil
import kagglehub

def main():
	dataset_slug = "acseckn/fabricnet"
	path = kagglehub.dataset_download(dataset_slug)
	print("Downloaded FabricNet via KaggleHub:", path)

	# Project data directory (where the notebook expects data)
	project_root = os.path.dirname(os.path.abspath(__file__))
	data_dir = os.path.join(project_root, "data")
	os.makedirs(data_dir, exist_ok=True)

	# Copy all files/folders from KaggleHub path into our project's data directory
	# This preserves original structure and makes DATA_ROOT usable as project-local path.
	copied = 0
	for name in os.listdir(path):
		src = os.path.join(path, name)
		dst = os.path.join(data_dir, name)
		try:
			if os.path.isdir(src):
				shutil.copytree(src, dst, dirs_exist_ok=True)
			else:
				shutil.copy2(src, dst)
			copied += 1
		except Exception as e:
			print(f"Warning: failed to copy {src} -> {dst}: {e}")

	print(f"Copied {copied} items into:", data_dir)
	print("Set DATA_ROOT to this path in the notebook:", data_dir)

if __name__ == "__main__":
	main()