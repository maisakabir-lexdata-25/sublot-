from ultralytics import YOLO
import yaml
from pathlib import Path

def run_data_yaml():
    output = []
    output.append("--- Running data.yaml Check ---")
    data_path = Path("data.yaml")
    
    if not data_path.exists():
        output.append(f"Error: {data_path} not found.")
    else:
        # Load and print content
        with open(data_path, 'r') as f:
            config = yaml.safe_load(f)
            output.append("YAML Content Loaded:")
            output.append(yaml.dump(config, default_flow_style=False))
        
        output.append("\nValidating dataset structure...")
        base_path = Path(config.get('path', '.'))
        for split in ['train', 'val', 'test']:
            split_p = base_path / config.get(split, 'missing')
            if split_p.exists():
                output.append(f"✓ {split} path found: {split_p}")
            else:
                output.append(f"✗ {split} path NOT found: {split_p}")

    output.append("\n--- data.yaml Run Complete ---")
    
    with open("data_run_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output))
    print("Results written to data_run_results.txt")

if __name__ == "__main__":
    run_data_yaml()
