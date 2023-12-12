import tarfile
import os

def h5_to_tar_gz(h5_file, output_path):
  """
  Converts a .h5 file to a .tar.gz archive.

  Args:
    h5_file: Path to the input .h5 file.
    output_path: Path to the output .tar.gz file.
  """
  with tarfile.open(output_path, mode="w:gz") as tar:
    tar.add(h5_file, arcname=os.path.basename(h5_file))

# Replace these values with your actual file paths
h5_file = "/Users/rohan_mehta/Documents/GitHub/MSOL-Capstone/lstm_base_2023.h5"
output_path = "/Users/rohan_mehta/Documents/GitHub/MSOL-Capstone/lstm_base_2023.tar.gz"

h5_to_tar_gz(h5_file, output_path)

print(f"Converted .h5 file to .tar.gz successfully: {output_path}")
