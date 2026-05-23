# 任务6 同genre分类标注规则
TASK6_ANNOTATION_RULES = '''## Core Strict Annotation Rules (Must follow strictly and keep consistent with the below examples)
1. Do NOT judge semantic similarity, factual correctness, viewpoint opposition, emotional contrast, or numerical differences.
2. The only judgment standard: Check if both sentences belong to the specified Genre, and focus on **the same core subject / same event / same sub-topic**.
3. If they belong to the same Genre and same topic, even with opposite descriptions, conflicting details, contradictory viewpoints or inconsistent numbers, mark Y uniformly.
4. Only mark N when the two sentences have no common subject, no related topic, and completely unrelated across fields.
5. If they only belong to the same broad Genre but have totally irrelevant sub-topics, directly mark N.

## Few-Shot Standard Examples (Strictly follow this annotation logic)
### Example 1【telephone - Opposite opinion still mark Y】
Input: Sentence 1: all right i agree with that people that are uh driving Sentence 2: People don't drive. Genre: telephone.
Output: Y

### Example 2【telephone - Irrelevant topic mark N】
Input: Sentence 1: whether that would uh get them addicted or not you know that that's a real worry Sentence 2: Shiva's symbol is in the east. Genre: telephone.
Output: N

### Example 3【travel - Contrary fact still mark Y】
Input: Sentence 1: Built in the third century a.d. they played host to the highly ranked of Roman Alexandria who would come to the baths to relax and exchange news while enjoying a soak or a massage. Sentence 2: There were no massages at the baths. Genre: travel.
Output: Y

### Example 4【travel - Same genre but different topic mark N】
Input: Sentence 1: Toward the southern end of the Dead Sea, a major health spa center has developed, with luxury and moderate hotels offering unique Dead Sea programs for relaxation, health, and beauty. Sentence 2: Hokkaido has a lot of fun winter sports that people drive for hours to take part in. Genre: travel.
Output: N

### Example 5【government - Opposite conclusion still mark Y】
Input: Sentence 1: These instances have been the subject of case studies. Sentence 2: These instances have not been studied. Genre: government.
Output: Y

### Example 6【government - Irrelevant field mark N】
Input: Sentence 1: The Federal Managers' Financial Integrity Act of 1982 requires agency management to annually assess and report on the adequacy of internal control. Sentence 2: The painter's date of birth is known, even though nobody knows who the painter is himself. Genre: government.
Output: N

### Example 7【fiction - Opposite character description still mark Y】
Input: Sentence 1: She had the transparent skin and classic features that occur once in a million times but which still keep the legend of redheaded enchantresses alive. Sentence 2: She had dark, tanned skin and light blonde hair. Genre: fiction.
Output: Y

### Example 8【fiction - Irrelevant plot mark N】
Input: Sentence 1: Yes, said Sir James gravely. Sentence 2: I know what you did. Genre: fiction.
Output: N

### Example 9【slate - Same current affairs topic mark Y】
Input: Sentence 1: If Finkelstein were to apply his logic to Lee Atwater's Willie Horton strategy, he'd have to write, Not race but crime served as the prime scapegoat of George Bush's 1988 campaign. Sentence 2: Blaming minorities was George Bush's main reason for problems during his 1988 campaign. Genre: slate.
Output: Y

### Example 10【slate - Irrelevant topic mark N】
Input: Sentence 1: Lifeboats have been constructed for the top dozen employees. Sentence 2: There is a chance that he does not know that it may be a cult because they have brainwashed him. Genre: slate.
Output: N'''


RULES = {
  1:'''## Rules:
1. Use tools
2. **Trust the result of the tool**''',

  3:'''## Rules:
1. Use tools
2. **Trust the result of the tool**''',

  4:'''## Rules:
1. Use tools
2. **Trust the result of the tool**''',
    7:'''Follow these steps to reason and answer:
1. Determine the domain from the given Category to narrow down the scope.
2. Extract key information from the Clue: proper nouns, years, locations, people, abbreviations, allusions, features and foreign language content.
3. Match the clues with general and encyclopedic knowledge to find the only correct answer.
4. Check if the answer matches the Category and rule out irrelevant options.
''',
2:'''
## Rules:
1. Use tools
2. **Trust the result of the tool**

## Example

### Example 1

Input: Sentence: 'The ladder of a jet is lowered from the side for loading passengers'. Count the number of verbs.
step 1: use count_verbs tool
step 2: output based on the tool result
<label>2</label>

### Example 2
Input: Sentence: 'The ladder of a jet is lowered from the side for loading passengers'. Count the number of nouns.
step 1: use count_nouns tool
step 2: output based on the tool result
<label>4</label>


''',
5:'''
## Rules:
**Think before you answer**
Combine full tweet text, hashtags and emojis for judgment.
Do NOT make judgment only based on emojis or hashtags.
Focus on the author’s real emotion and context in the text.
Sad could also means angry, discouraged， guilty， awkward，critize, complain and so on
Output label strictly: only Sad or Not sad.

## Reasoning & Labeling Process
Step 1: Analyze the author’s emotion from context
Step 2: Check emojis/hashtags as auxiliary clues
Step 3: Make a comprehensive analysis.
Step 4: Output final label: Sad / Not sad

## Few-shot Examples (By Pattern)

### Example 1
- Neutral sentence with no sad emoji. == Not Sad
Input: @badpostyoongi I know for a fact they'll either ignore the fact tiff isn't or change Cindy's background
<label>Not sad</label>

### Example 2
- Negative statement with a sad emoji. == Sad
Input: Went to bed a 1:30, fell asleep after, my niece started crying at 4. I'm dying... 😧
<label>Sad</label>

### Example 3
- Positive/playful statement with a sad emoji (sad emoji for joking). == Not Sad
Input: @bxchpls03 U so lucky ahu 😭
<label>Not sad</label>

### Example 4
- Negative complaint statement with no emoji. == Sad
Input: @RealSkipBayless Your opinions on sports is dreadful
<label>Sad</label>

### Example 5
- Negative statement but for joking, no real sadness. == Not Sad
Input: Do people notice that only saying 'You're so pretty' when I have make-up on. Is offense! \n& I take note that they never say it when I don't.
<label>Not sad</label>

### Example 6
- Short negative sentence with only sad emoji. == Sad
Input: Same ☹
<label>Sad</label>

'''
}
