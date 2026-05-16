import logging

logger = logging.getLogger("COG-6")

class PromptBuilder:
    def __init__(self):
        pass

    def build_prompt(self, definition: str, examples: list, query: str) -> str:
        logger.info("[COG-6] Prompt assembled using adaptive context")
        
        prompt = ""
        if definition:
            prompt += f"Task Definition:\n{definition}\n\n"
        
        if examples:
            prompt += "Examples:\n"
            for i, ex in enumerate(examples):
                prompt += f"Example {i+1}:\n"
                prompt += f"Input: {ex.get('input', '')}\n"
                
                output_val = ex.get('output', '')
                if isinstance(output_val, list) and len(output_val) > 0:
                    output_val = output_val[0]
                prompt += f"Output: {output_val}\n\n"
        
        prompt += f"Now solve the following task:\nInput: {query}\nOutput:"
        
        return prompt
