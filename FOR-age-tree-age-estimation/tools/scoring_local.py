import json
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import argparse

# ------------- Argument Parser -------------
parser = argparse.ArgumentParser(description='Evaluate tree species classification results.')
parser.add_argument('--pred', type=str, required=True, help='Path to prediction CSV file')
parser.add_argument('--gt', type=str, required=True, help='Path to ground truth CSV file')
parser.add_argument('--out', type=str, required=False, default='.', help='Output directory')
args = parser.parse_args()

csv_file = args.pred
truth_file = args.gt
score_dir = args.out
html_file = os.path.join(score_dir, 'detailed_results.html')


# ------------- Utility Functions -------------
def write_file(file, content):
    with open(file, 'a', encoding="utf-8") as f:
        f.write(content)

def plot_confusion(y_true, y_pred, coniferous_species):
    all_species = sorted(list(set(y_true) | set(y_pred)))
    broadleaves_species = sorted([s for s in all_species if s not in coniferous_species])
    species_labels = coniferous_species + broadleaves_species

    conf_matrix = confusion_matrix(y_true, y_pred, labels=species_labels)
    norm_conf_matrix = np.round((conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]) * 100, 0)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(norm_conf_matrix, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=species_labels, yticklabels=species_labels, ax=ax)
    plt.title('Normalized (%) Matrix for Tree Species Prediction')
    plt.xlabel('Predicted Species')
    plt.ylabel('Reference Species')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)
    return fig

def plot_accuracy_by_data_type(merged_df, data_type_col='data_type', y_true_col='species', y_pred_col='predicted_species'):
    accuracy_by_type = merged_df.groupby(data_type_col).apply(
        lambda df: accuracy_score(df[y_true_col], df[y_pred_col])
    ).reset_index()
    accuracy_by_type.columns = [data_type_col, 'accuracy']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(accuracy_by_type[data_type_col], accuracy_by_type['accuracy'], color='skyblue')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    ax.set_title('Overall Accuracy by Data Type')
    ax.set_xlabel('Data Type')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')


# ------------- Scoring Logic -------------

# Load CSV files
print("Reading prediction:", csv_file)
predictions = pd.read_csv(csv_file)

print("Reading ground truth:", truth_file)
truth = pd.read_csv(truth_file)

# Merge on 'treeID'
merged_df = pd.merge(predictions, truth, on='treeID')

y_true = merged_df['species']
y_pred = merged_df['predicted_species']

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

scores = {
    'OA': round(accuracy, 2),
    'P': round(precision, 2),
    'R': round(recall, 2),
    'F1': round(f1, 2)
}

# Save scores
print("Scores:", scores)
with open(os.path.join(score_dir, 'scores.json'), 'w') as f:
    json.dump(scores, f)

# Generate HTML report
coniferous_species = [
     "Abies_alba","Larix_decidua", "Picea_abies", "Picea_glauca", 
     "Pinus_contorta", "Pinus_nigra", "Pinus_pinaster", "Pinus_radiata","Pinus_resinosa",
     "Pinus_sylvestris","Pseudotsuga_menziesii"
]

# 1. Confusion Matrix
conf_fig = plot_confusion(y_true, y_pred, coniferous_species)
conf_b64 = fig_to_b64(conf_fig)
write_file(html_file, f'<h2>Confusion Matrix</h2><img src="data:image/png;base64,{conf_b64}"><br>')

# 2. Accuracy by Data Type (if exists)
if 'data_type' in merged_df.columns:
    bar_fig = plot_accuracy_by_data_type(merged_df)
    bar_b64 = fig_to_b64(bar_fig)
    write_file(html_file, f'<h2>Accuracy by Data Type</h2><img src="data:image/png;base64,{bar_b64}"><br>')
else:
    print("Warning: 'data_type' column not found; skipping accuracy by type plot.")
