---
name: sadness_detection
description: Judge whether the author of the tweet is sad or not. 
always: true
---

# Rules
Combine full tweet text, hashtags and emojis for judgment.
Do NOT make judgment only based on emojis or hashtags.
Focus on the author’s real emotion and context in the text.
Output label strictly: only Sad or Not sad.

# Reasoning & Labeling Process
Step 1: Analyze the author’s emotion from context
Step 2: Check emojis/hashtags as auxiliary clues
Step 3: Make a comprehensive analysis, sometime an sad emoji means happiness and vice versa.
Step 4: Output final label: Sad / Not sad

# Few-shot Examples (By Pattern)

## Example 1
- Neutral sentence with no sad emoji. == Not Sad
Input: @badpostyoongi I know for a fact they'll either ignore the fact tiff isn't or change Cindy's background
Step 1: Tweet: @badpostyoongi I know for a fact they'll either ignore the fact tiff isn't or change Cindy's background
Step 2: Text shows discussion and speculation, no sad emotion
Step 3: No sad emojis or hashtags
Step 4: Label: Not sad
Output: <label>Not sad</label>

## Example 2
- Negative statement with a sad emoji. == Sad
Input: Went to bed a 1:30, fell asleep after, my niece started crying at 4. I'm dying... 😧
Step 1: Tweet: Went to bed a 1:30, fell asleep after, my niece started crying at 4. I'm dying... 😧
Step 2: Text shows tired, uncomfortable and bad mood
Step 3: 😧 is a sad emoji that supports sad emotion
Step 4: Label: Sad
Output: <label>Sad</label>

## Example 3
- Positive/playful statement with a sad emoji (sad emoji for joking). == Not Sad
Input: @bxchpls03 U so lucky ahu 😭
Step 1: Tweet: @bxchpls03 U so lucky ahu 😭
Step 2: Text expresses playful envy, no real sadness
Step 3: 😭 is used for playful emotion, not real sorrow
Step 4: Label: Not sad
Output: <label>Not sad</label>

## Example 4
- Negative complaint statement with no emoji. == Sad
Input: @RealSkipBayless Your opinions on sports is dreadful
Step 1: Tweet: @RealSkipBayless Your opinions on sports is dreadful
Step 2: Text expresses strong negative and unpleasant feeling
Step 3: No emoji or hashtag
Step 4: Label: Sad
Output: <label>Sad</label>

## Example 5
- Negative statement but for joking, no real sadness. == Not Sad
Input: Do people notice that only saying 'You're so pretty' when I have make-up on. Is offense! \n& I take note that they never say it when I don't.
Step 1: Tweet: Do people notice that only saying 'You're so pretty' when I have make-up on. Is offense! \n& I take note that they never say it when I don't.
Step 2: Text expresses light dissatisfaction in a joking tone, no real sadness
Step 3: No emoji or hashtag
Step 4: Label: Not sad
Output: <label>Not sad</label>

## Example 6
- Short negative sentence with only sad emoji. == Sad
Input: Same ☹
Step 1: Tweet: Same ☹
Step 2: Text shows agreement with sad feeling
Step 3: ☹ is a clear sad emoji
Step 4: Label: Sad
Output: <label>Sad</label>