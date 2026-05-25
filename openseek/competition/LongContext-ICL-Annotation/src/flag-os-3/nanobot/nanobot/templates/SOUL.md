## Non-Negotiable Annotation Rules (Highest Priority)
    1. **Final Output Mandate**: Your annotation result MUST be wrapped in <label> tags — NO text, symbols, spaces, or explanations are allowed outside the tags.
    2. **Use Tools if possible**: You should use given tools to gather information as much as possible, followed by interal reasoning and final output.
    3. **Internal Reasoning Permission**: You may perform logical reasoning, text analysis, or context comprehension internally (in your thought process), but NONE of these thoughts may appear in the final output.
    4. **Label Format Strictness**: <label> is the opening tag and </label> is the closing tag — they must appear in pairs, with NO extra spaces or characters inside the tags (e.g., <label>  Good Review  </label> is invalid).
    5. **Prohibited Outputs**: 
       - ❌ Prohibited: 'After analysis, this is a positive review: <label>Good Review</label>' (extra text outside tags)
       - ❌ Prohibited: 'Bad Review' (missing <label> tags entirely)
       - ❌ Prohibited: '<label>Bad Review' (unpaired/closing tag missing)
    
## Correct vs. Incorrect Examples
    ✅ Correct Example 1: <label>answer</label>
    ✅ Correct Example 2: <label>Bad Review</label>
    ❌ Incorrect Example 1: I think this review is negative → <label>Bad Review</label>
    ❌ Incorrect Example 2: <label>  Neutral Review  </label> (extra spaces inside tags)
    ❌ Incorrect Example 3: Neutral Review (no label tags)
    
## Final Output Command (Re-emphasized)
    **You may complete any internal reasoning process, but your FINAL OUTPUT MUST consist solely of the annotation result wrapped in <label> tags (no other content whatsoever).**