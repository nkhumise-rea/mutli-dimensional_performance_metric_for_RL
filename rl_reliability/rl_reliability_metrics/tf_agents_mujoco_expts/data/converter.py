import pandas as pd


algo = 'sac' #'ddpg' #'td3' #
for i in range(1,7):
    input_csv = f"csv_data_final/{algo}_reach_{i}_train.csv"
    output_csv = f"csv_data/{algo}_reach_{i}_train.csv"

    column_to_delete = "Wall time"
    new_column_names = {
        "Step":"Metrics/EnvironmentSteps",
        "Value":"Metrics/AverageReturn"
    }

    df = pd.read_csv(input_csv)

    df = df.drop(columns=[column_to_delete])

    df = df.rename(columns=new_column_names)

    df.to_csv(output_csv, index=False)

print("conversion complete. SAved to: ", output_csv)