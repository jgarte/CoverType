import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score

output_file_path = Path('data/output/')
validation_file_path = Path('data/validation')
submission_file = Path(sys.argv[1])
validation_file = Path('validation.csv')

submission = pd.read_csv(output_file_path/submission_file, usecols=['Cover_Type'])
validation = pd.read_csv(validation_file_path/validation_file, usecols=['Cover_Type'])

score = accuracy_score(validation['Cover_Type'].values, submission['Cover_Type'].values)

print('{} accuracy score: {}'.format(submission_file, round(score, 8)))

