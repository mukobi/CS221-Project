import csv
import os

SUMMARY_FILENAME = 'experiments_summary.csv'

# declare important file paths
PROJECT_PATH = os.path.abspath('')
EXPERIMENTS_PATH = PROJECT_PATH + '/experiments/'
SUMMARY_PATH = PROJECT_PATH + '/' + SUMMARY_FILENAME

# experiment file schema
EXPERIMENT_COLUMNS = ['time (m)', 'loss', 'train accuracy', 'val accuracy']
VAL_ACCURACY_INDEX = EXPERIMENT_COLUMNS.index('val accuracy')

# collect performance of best model from each experiment
experiments_summary = []
num_parsed_files = 0
for experiment in os.listdir(EXPERIMENTS_PATH):
    if '.csv' not in experiment:
        continue
    best_model_row = None
    with open(EXPERIMENTS_PATH + experiment, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i > 0:
                # compare val accuracy and update best row
                val_accuracy = row[VAL_ACCURACY_INDEX]
                if not best_model_row or val_accuracy > best_model_row[VAL_ACCURACY_INDEX]:
                    best_model_row = row
        print(f'Processed {i} lines for {experiment}.')
    # print(f'Best row: {best_model_row}')
    experiments_summary.append([experiment] + best_model_row)
    num_parsed_files += 1

print('Parsed {} files.'.format(num_parsed_files))

# write it all to one csv file
with open(SUMMARY_PATH, 'w', newline='') as csv_file:
    csv_writer = csv.writer(
        csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['experiment name'] + EXPERIMENT_COLUMNS)
    csv_writer.writerows(experiments_summary)
