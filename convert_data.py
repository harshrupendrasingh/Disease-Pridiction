import pandas as pd

data_file = 'chatbot\Dataset\disease symptoms\merged_data.csv'
merged_data = pd.read_csv(data_file)
print(merged_data.columns)
