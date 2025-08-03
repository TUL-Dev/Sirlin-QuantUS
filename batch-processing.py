import yaml
from pathlib import Path

from quantus.full_workflow import main_dict

def analyze_file(config_dict: dict, scan_path: str, 
                 phantom_path: str, seg_path: str) -> int:
    """Runs the full QuantUS workflow from a config dictionary and file paths.
    
    Args:
        config_dict (dict): Configuration dictionary with all necessary parameters.
        scan_path (str): Path to the ultrasound scan file.
        phantom_path (str): Path to the phantom file.
        seg_path (str): Path to the segmentation file.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """

    # Update the config dictionary with file paths
    config_dict['scan_path'] = scan_path
    config_dict['phantom_path'] = phantom_path
    config_dict['seg_path'] = seg_path

    # Run the core pipeline with the updated config
    return main_dict(config_dict)

def analyze_dataset(yaml_path: str, dataset_path: str) -> int:
    """Runs the full QuantUS workflow through an entire dataset from a config dictionary and dataset path.
    Iterates through custom dataset file directory structure to find scan, phantom, and segmentation files.

    Args:
        yaml_path (str): Path to the YAML configuration file.
        dataset_path (str): Path to the dataset directory containing scan, phantom, and segmentation files.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Folder initializations
    phantom_folder = Path(dataset_path) / "_Phantom_Exams" / "Calibration_Phantoms" / "CL"
    results_dir = Path(dataset_path) / 'QuantUS Analysis Results'
    results_dir.mkdir(exist_ok=True)

    for patient_folder in Path(dataset_path).iterdir():
        if not patient_folder.name.startswith('_') and not patient_folder.name.startswith('.') \
              and patient_folder.is_dir() and patient_folder != results_dir:
            scan_num = patient_folder.name[-3:]
            exam_folders = patient_folder / "Sorted"

            for exam_folder in exam_folders.iterdir():
                if exam_folder.is_dir() and exam_folder.name.startswith("PQ_"):
                    for scan_folder in exam_folder.iterdir():
                        if scan_folder.is_dir() and scan_folder.name.startswith("Image"):
                            for scan_path in scan_folder.iterdir():
                                if scan_path.name.endswith('.tar') and not scan_path.name.startswith('.'):
                                    # Construct paths for scan, phantom, and segmentation files
                                    phantom_path = phantom_folder / f"P19_PQ_{scan_num}_CL" / "Image 1" / "QUS1.tar"
                                    seg_path = scan_path.with_suffix('.pkl')
                                    assert scan_path.exists() and phantom_path.exists()
                                    if not seg_path.exists():
                                        print(f"Segmentation file not found for PQ_{scan_num}/{scan_path.parents[1].name}/{scan_path.parent.name}/{scan_path.name}, skipping...")
                                        continue
                                    assert scan_path.exists() and phantom_path.exists() and seg_path.exists()
                                    config_dict['scan_path'] = str(scan_path)
                                    config_dict['phantom_path'] = str(phantom_path)
                                    config_dict['seg_path'] = str(seg_path)

                                    # Set paths for output files
                                    out_dir = results_dir / patient_folder.name / exam_folder.name / scan_folder.name
                                    out_dir.mkdir(parents=True, exist_ok=True)
                                    config_dict['visualization_kwargs']['paramap_folder_path'] = str(out_dir)
                                    config_dict['data_export_path'] = str(out_dir / 'results.csv')

                                    if exit_code := main_dict(config_dict):
                                        return exit_code

    combine_numerical_results(results_dir)
    return 0

def combine_numerical_results(results_dir: Path) -> None:
    """Combines numerical results from all processed scans into a single CSV file.
    
    Args:
        results_dir (Path): Path to the directory containing the results of the analysis.
    """
    all_results = []
    for result_file in results_dir.glob('**/*.csv'):
        if result_file.name == 'combined_results.csv':
            raise ValueError("Cannot combine results with the same name as the output file. Either delete or rename 'combined_results.csv' before running this function.")
        df = pd.read_csv(result_file)
        
        all_results.append(df)

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(results_dir / 'combined_results.csv', index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch process QuantUS analysis on a dataset.")
    parser.add_argument('yaml_path', type=str, help='Path to the YAML configuration file.')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset directory.')
    
    args = parser.parse_args()
    
    exit_code = analyze_dataset(args.yaml_path, args.dataset_path)
    if exit_code == 0:
        print("Batch processing completed successfully.")
    else:
        print(f"Batch processing failed with exit code {exit_code}.")
    exit(exit_code)
