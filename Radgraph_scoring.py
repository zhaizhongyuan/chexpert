import pandas as pd
import json
from radgraph import F1RadGraph
import os
from tqdm import tqdm

# Configuration - Edit these variables as needed
GROUND_TRUTH_CSV = "./ground_truth.csv"  # Path to CSV file containing ground truth reports
GENERATED_CSV = "./generated_reports.csv"  # Path to CSV file containing generated reports
OUTPUT_JSON = "radgraph_scores.json"  # Path to output JSON file
OUTPUT_CSV = "radgraph_scores.csv"  # Path to output CSV file
MODEL_TYPE = "radgraph"  # Options: 'radgraph', 'radgraph-xl', 'echograph'
REWARD_LEVEL = "simple"  # Options: 'simple', 'partial', 'complete', 'all'
BATCH_SIZE = 1  # Batch size for processing

def main():
    # Load CSV files
    print(f"Loading ground truth reports from {GROUND_TRUTH_CSV}")
    ground_truth_df = pd.read_csv(GROUND_TRUTH_CSV)
    
    print(f"Loading generated reports from {GENERATED_CSV}")
    generated_df = pd.read_csv(GENERATED_CSV)
    
    # Ensure both dataframes have the required columns
    required_columns = ['report', 'study_id']
    for df, name in [(ground_truth_df, 'ground truth'), (generated_df, 'generated')]:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {name} dataframe: {missing}")
    
    # Merge dataframes on study_id to ensure we're comparing the same studies
    print("Merging datasets on study_id")
    merged_df = pd.merge(ground_truth_df, generated_df, on='study_id', suffixes=('_gt', '_gen'))
    
    print(f"Found {len(merged_df)} matching reports")
    
    # Initialize F1RadGraph
    print(f"Initializing F1RadGraph with model_type={MODEL_TYPE}, reward_level={REWARD_LEVEL}")
    f1radgraph = F1RadGraph(
        reward_level=REWARD_LEVEL,
        model_type=MODEL_TYPE,
        batch_size=BATCH_SIZE
    )
    
    # Extract reports
    references = merged_df['report_gt'].tolist()
    hypotheses = merged_df['report_gen'].tolist()
    study_ids = merged_df['study_id'].tolist()
    
    # Run F1RadGraph
    print("Computing scores...")
    mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(
        refs=references,
        hyps=hypotheses
    )
    
    # Prepare results
    if REWARD_LEVEL == 'all':
        # For 'all' reward level, we get precision, recall, and F1
        result = {
            'overall': {
                'precision': float(mean_reward[0]),
                'recall': float(mean_reward[1]),
                'f1': float(mean_reward[2])
            },
            'by_report': []
        }
        
        # Create lists for CSV export
        csv_data = {
            'study_id': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        # Add per-report scores
        for i, study_id in enumerate(study_ids):
            result['by_report'].append({
                'study_id': study_id,
                'ground_truth': references[i],
                'generated': hypotheses[i],
                'precision': float(reward_list[0][i]),
                'recall': float(reward_list[1][i]),
                'f1': float(reward_list[2][i]),
                'ground_truth_annotations': reference_annotation_lists[i] if i < len(reference_annotation_lists) else None,
                'generated_annotations': hypothesis_annotation_lists[i] if i < len(hypothesis_annotation_lists) else None
            })
            
            # Add to CSV data
            csv_data['study_id'].append(study_id)
            csv_data['precision'].append(float(reward_list[0][i]))
            csv_data['recall'].append(float(reward_list[1][i]))
            csv_data['f1'].append(float(reward_list[2][i]))
            
    else:
        # For other reward levels, we get a single score
        result = {
            'overall': {
                'score': float(mean_reward)
            },
            'by_report': []
        }
        
        # Create lists for CSV export
        csv_data = {
            'study_id': [],
            'score': []
        }
        
        # Add per-report scores
        for i, study_id in enumerate(study_ids):
            result['by_report'].append({
                'study_id': study_id,
                'ground_truth': references[i],
                'generated': hypotheses[i],
                'score': float(reward_list[i]),
                'ground_truth_annotations': reference_annotation_lists[i] if i < len(reference_annotation_lists) else None,
                'generated_annotations': hypothesis_annotation_lists[i] if i < len(hypothesis_annotation_lists) else None
            })
            
            # Add to CSV data
            csv_data['study_id'].append(study_id)
            csv_data['score'].append(float(reward_list[i]))
    
    # Save JSON results
    print(f"Saving detailed results to {OUTPUT_JSON}")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Save CSV results
    print(f"Saving scores to {OUTPUT_CSV}")
    scores_df = pd.DataFrame(csv_data)
    scores_df.to_csv(OUTPUT_CSV, index=False)
    
    # Print summary
    print("\nScoring Summary:")
    if REWARD_LEVEL == 'all':
        print(f"Overall Precision: {result['overall']['precision']:.4f}")
        print(f"Overall Recall: {result['overall']['recall']:.4f}")
        print(f"Overall F1: {result['overall']['f1']:.4f}")
    else:
        print(f"Overall Score: {result['overall']['score']:.4f}")
    
    print(f"\nDetailed results saved to {OUTPUT_JSON}")
    print(f"Score metrics saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()