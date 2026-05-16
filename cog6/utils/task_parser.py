import json
import logging
from typing import Dict, Any

logger = logging.getLogger("COG-6")

class TaskParser:
    def __init__(self):
        logger.info("[COG-6] TaskParser initialized")

    def parse(self, filepath: str) -> Dict[str, Any]:
        logger.info(f"[COG-6] Parsing task file: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        task_id = data.get("task_id", "unknown_task")
        task_name = data.get("task_name", "unknown")
        definition = data.get("Definition", [""])[0]
        examples = data.get("examples", [])
        
        # Determine valid labels by inspecting the examples' output values
        # OpenSeek outputs are typically arrays like ["Sad"]
        labels_set = set()
        for ex in examples:
            outputs = ex.get("output", [])
            if isinstance(outputs, list):
                for out in outputs:
                    if isinstance(out, str):
                        labels_set.add(out)
            elif isinstance(outputs, str):
                labels_set.add(outputs)
                
        valid_labels = list(labels_set)
        
        logger.info(f"[COG-6] Parsed Task Labels:\n{valid_labels}")
        
        return {
            "task_id": task_id,
            "task_name": task_name,
            "definition": definition,
            "examples": examples,
            "valid_labels": valid_labels
        }
