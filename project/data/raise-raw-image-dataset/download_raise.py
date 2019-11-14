"""Download images from urls in a .csv file."""

import csv
import os
import requests
import time

# local csv file with links to images from dataset (download from RAISE site)
RAISE_CSV_FILENAME = "raise.csv"
# column in RAISE_CSV_FILENAME with the names to download
NAME_COLUMN = "File"
# column in RAISE_CSV_FILENAME with the URLs to download
URL_COLUMN = "TIFF"
# file extension to give downloaded images
FILE_EXTENSION = ".TIF"
# folder in which to put downloaded images
SUBFOLDER = "/raw/"
# log info every nth file
N_LOG_INTERVAL = 100

FOLDER_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # load urls and names from csv
    full_csv_file_path = FOLDER_DIR + "/" + RAISE_CSV_FILENAME

    csv_data_dict = {}
    with open(full_csv_file_path, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader, None)  # read headers from first line
        for h in headers:
            csv_data_dict[h] = []
        for row in reader:
            for h, v in zip(headers, row):
                csv_data_dict[h].append(v)

    names_to_download = csv_data_dict[NAME_COLUMN]
    urls_to_download = csv_data_dict[URL_COLUMN]
    assert len(names_to_download) == len(urls_to_download)

    total_num_files = len(urls_to_download)
    print("Found a total of {} files to download.".format(total_num_files))

    # download files
    subfolder_full_path = FOLDER_DIR + SUBFOLDER
    if not os.path.exists(subfolder_full_path):
        os.mkdir(subfolder_full_path)

    start_time = time.time()
    for i in range(total_num_files):
        name = names_to_download[i]
        local_filepath = subfolder_full_path + name + FILE_EXTENSION
        url = urls_to_download[i]

        with open(local_filepath, 'wb') as handle:
            response = requests.get(url, stream=True)
            if not response.ok:
                print(response)
            for block in response.iter_content(1024):
                if not block:
                    break
                handle.write(block)

        if i + 1 % N_LOG_INTERVAL == 0:
            print("Downloaded {:}/{:} images in {:.2f} minutes".format(
                i + 1, total_num_files, (time.time() - start_time) / 60))

    print("Downloaded all images!")


if __name__ == "__main__":
    main()
