import logging

logger = logging.getLogger("COG-6")

class TaskRouter:
    def __init__(self):
        logger.info("[COG-6] Task-Aware Retrieval Router initialized")

    def get_strategy(self, task_name: str) -> str:
        task_name = str(task_name).lower()
        
        semantic_keywords = ["sentiment", "qa", "mnli"]
        lexical_keywords = ["symbolic", "counting", "code"]
        
        for kw in lexical_keywords:
            if kw in task_name:
                logger.info("[COG-6] Retrieval strategy selected: lexical-heavy")
                return "lexical-heavy"
                
        for kw in semantic_keywords:
            if kw in task_name:
                logger.info("[COG-6] Retrieval strategy selected: semantic")
                return "semantic"
                
        # Default fallback
        logger.info("[COG-6] Retrieval strategy selected: semantic")
        return "semantic"
