import argparse
import urllib.request
import os
import tarfile
import sys

# Crea el objeto ArgumentParser
parser = argparse.ArgumentParser(description='Descripción del script')

# Agrega opciones
parser.add_argument('--txt', help='path to the .txt that contains all urls from dataset')
parser.add_argument('--n', help='number of subsets to download')
parser.add_argument('--outputPath', help='path to save the subdatasets')

# Analiza los argumentos de la línea de comandos
args = parser.parse_args()

# Accede a los valores de las opciones
if args.txt:
    print(f'Text file with urls : {args.txt}')
if args.n:
    print(f'Number of files to download: {args.n}')
if args.outputPath:
        print(f'Output path: {args.n}')
    
def progress_callback(count, block_size, total_size):
    downloaded_size = count * block_size
    downloaded_mb = downloaded_size / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    percent = int(downloaded_size * 100 / total_size)
   # sys.stdout.write(f"\rDownloaded: {downloaded_mb:.2f} MB / {total_mb:.2f} MB  ({percent}%)")
    sys.stdout.flush()

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Folder created:", path)
    else:
        print("Folder exists:", path)
        
path="subDataset"     
current_dir = os.getcwd()        
subDataset = os.path.join(current_dir, path)
split = os.path.join(subDataset, args.outputPath)
#location for downloading tarz files
create_folder(subDataset)
#location to extract tarz
create_folder(split)
print(subDataset)
print(split)


# Open the file in read mode
with open(args.txt, "r") as file:
    line_count = 0  # Counter variable for lines
    max_lines = int(args.n)  # Number of lines to read
    # Iterate over each line in the file
    for line in file:
        # Do something with each line
        download_path = os.path.join(subDataset, "Subset_" + str(line_count)+".tar")
        if line_count >51:
          urllib.request.urlretrieve(line.strip(),download_path,  reporthook=progress_callback)
          print("File downloaded successfully:" + str(line_count))
          
        line_count += 1  # Increment the line counter

        # Check if the desired number of lines has been reached
        if line_count >= max_lines:
            break
        


# Get the current working directory and joing with the args.outputpath

#Iterate over all files in the current directory
for file_name in os.listdir(subDataset):
    file_path = os.path.join(subDataset, file_name)
    
    
        # Create a folder with the same name as the .tar file
    folder_name = os.path.splitext(file_name)[0]
    folder_path = os.path.join(split, folder_name)
    #os.mkdir(folder_path)
    # Check if the file is a .tar file
    if file_name.endswith(".tar"):
        print(f"Extracting {file_name}...")

        # Open the .tar file
        with tarfile.open(file_path, "r") as tar:
            # Extract all files in the .tar file
            tar.extractall(path=folder_path)
        
        print("Extraction complete.")

print("All .tar files extracted.")