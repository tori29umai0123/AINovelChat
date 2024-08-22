# How to use this file
#
# 1. create a folder called "hooks" in your repo
# 2. copy this file there
# 3. add the --additional-hooks-dir flag to your pyinstaller command:
#    ex: `pyinstaller --name binary-name --additional-hooks-dir=./hooks entry-point.py`

from PyInstaller.utils.hooks import collect_data_files
import os
import sys

# Define the package name
package_name = 'llama_cpp'

# Get the current working directory
current_dir = os.path.dirname(__file__)

# Define relative paths to the library files
if os.name == 'nt':  # Windows
    dll_path = os.path.join(current_dir, '..', 'venv', 'Lib', 'site-packages', package_name, 'lib', 'llama.dll')
    datas = collect_data_files(package_name) + [(dll_path, 'llama_cpp')]
elif sys.platform == 'darwin':  # Mac
    so_path = os.path.join(current_dir, '..', 'venv', 'Lib', 'site-packages', package_name, 'lib', 'llama.dylib')
    datas = collect_data_files(package_name) + [(so_path, 'llama_cpp')]
elif os.name == 'posix':  # Linux
    so_path = os.path.join(current_dir, '..', 'venv', 'Lib', 'site-packages', package_name, 'lib', 'libllama.so')
    datas = collect_data_files(package_name) + [(so_path, 'llama_cpp')]
else:
    datas = collect_data_files(package_name)

# Ensure to include the path to collect any other necessary data files
# if you have any other files to be included in the final bundle
