import pandas as pd

# Reading CSV files to generate lists of miRNA and disease names
miRNA_df = pd.read_csv('miRNA name.csv', header=None)
disease_df = pd.read_csv('disease number.csv', header=None)

# Removing headers (assuming the header is the first item) and generating lists
miRNAs = miRNA_df[0].tolist()[1:]  # Excluding the header
diseases = disease_df[0].tolist()[1:]  # Excluding the header

# Loading the association matrix Excel file
relation_matrix_df = pd.read_excel('average_score.xlsx', header=None)

# Creating a mapping from disease names to column indices
disease_to_index = {disease: idx for idx, disease in enumerate(diseases)}


# Defining a function to find associated miRNAs for a given disease name and write top 50 predictions to an Excel file
def find_associations_and_write_to_excel(disease_name):
    # Getting the column index for the given disease name
    disease_index = disease_to_index.get(disease_name)
    disease_index=disease_index+1
    if disease_index is None:
        print(f"Disease '{disease_name}' not found.")
        return

    # Getting the corresponding column data
    relation_column = relation_matrix_df.iloc[:, disease_index]

    # Finding all miRNAs with a value of 1
    known_associations_indices = relation_column[relation_column == 1].index.tolist()
    print("These elements are known associated samples:")
    for idx in known_associations_indices:
        print(miRNAs[idx])

    # Finding miRNAs with a value other than 1 and sorting them from highest to lowest, taking the top 50
    predictions_indices = relation_column[relation_column != 1].sort_values(ascending=False).head(50).index.tolist()
    print("\nTop 50 predicted miRNAs:")
    predictions = []
    for rank, idx in enumerate(predictions_indices, 1):
        miRNA_name = miRNAs[idx]
        predicted_value = relation_column.iloc[idx]
        predictions.append((f"Rank {rank} prediction", miRNA_name, predicted_value))
        print(f"Rank {rank} prediction: {miRNA_name}, Predicted value: {predicted_value}")

    # Write the predictions to an Excel file if any predictions are found
    if predictions:
        predictions_df = pd.DataFrame(predictions, columns=['Rank', 'miRNA', 'Predicted Value'])
        output_filename = f"{disease_name.replace(' ', '_')}_top_50_predictions.xlsx"
        predictions_df.to_excel(output_filename, index=False)
        print(f"Predictions written to {output_filename}")


# Example: Finding related miRNAs for "Lung Neoplasms" and writing the top 50 predictions to an Excel file
find_associations_and_write_to_excel('Carcinoma, Hepatocellular')

