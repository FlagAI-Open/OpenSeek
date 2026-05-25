import json
import zipfile
from pathlib import Path

def ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def get_output_file(task_id: int | str, output_dir: Path, split: str = 'test_samples') -> Path:
    return output_dir / f'openseek-{task_id}-v1.jsonl'

def save_jsonl(records: list[dict], output_file: Path):
    with output_file.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def fill_missing_submissions(output_dir: Path, data_dir: Path):
    for i in range(1, 9):
        file_path = get_output_file(i, output_dir)
        if not file_path.exists():
            records = []
            pattern = f'openseek-{i}_*.json'
            data_files = list(data_dir.glob(pattern))
            if data_files:
                try:
                    with data_files[0].open('r', encoding='utf-8') as f:
                        data = json.load(f)
                        for sample in data.get('test_samples', []):
                            records.append({'test_sample_id': sample['id'], 'prediction': ''})
                except Exception:
                    pass
            if records:
                save_jsonl(records, file_path)

def zip_submission_files(output_dir: Path, zip_path: Path, data_dir: Path):
    ensure_directory(output_dir)
    ensure_directory(zip_path.parent)
    fill_missing_submissions(output_dir, data_dir)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in sorted(output_dir.glob('*.jsonl')):
            zipf.write(f, f.name)
