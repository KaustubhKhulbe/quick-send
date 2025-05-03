import pandas as pd
import os

data_dir = "./compress_results/"  # <-- Change this

files = [f for f in os.listdir(data_dir) if f.endswith(".csv") or f.endswith(".txt")]

df_list = []
for file in files:
    file_path = os.path.join(data_dir, file)
    
    try:
        df = pd.read_csv(file_path, sep="\t|,", engine="python")  # Handles both comma and tab separation
        image_name = os.path.splitext(file)[0]  # Get the base name without extension
        image_name = image_name.replace("results_", "")
        df["Image"] = image_name # Add filename as a new column
        df_list.append(df)
    except Exception as e:
        print(f"Failed to load {file}: {e}")

# Concatenate all dataframes into one
final_df = pd.concat(df_list, ignore_index=True)

# Optional: Save it to a CSV
final_df.to_csv("consolidated_output.csv", index=False)

# Print the first few rows
print(final_df.head())
