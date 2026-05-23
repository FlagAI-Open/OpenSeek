---
name: count_nouns_verbs
description: Count the number of nouns/verbs in the given sentence.
always: true
---

# Count Nouns or Verbs

## Rules:
1. For verbs: Only count real content verbs. Exclude is/are/being etc.
2. For nouns: Count common nouns as in examples.
3. Steps:
   Step 1: Split into words
   Step 2: Label in ONE line: word(tag) word(tag) ...
   Step 3: Count and give final number
4. Output Final result

## Example

### Example 1

Input: Sentence: 'The ladder of a jet is lowered from the side for loading passengers'. Count the number of verbs.

Step 1: The, ladder, of, a, jet, is, lowered, from, the, side, for, loading, passengers
Step 2: The(other) ladder(other) of(other) a(other) jet(other) is(other) lowered(verb) from(other) the(other) side(other) for(other) loading(verb) passengers(other)
Step 3: Number of verbs = 2

Output: <label>2</label>


### Example 2
Input: Sentence: 'Ironic picture of man and woman walking up a sidewalk under a "Wrong Way" sign'. Count the number of nouns.

Step 1: Ironic, picture, of, man, and, woman, walking, up, a, sidewalk, under, a, Wrong, Way, sign
Step 2: Ironic(other) picture(noun) of(other) man(noun) and(other) woman(noun) walking(other) up(other) a(other) sidewalk(noun) under(other) a(other) Wrong(noun) Way(noun) sign(noun)
Step 3: Number of nouns = 6
Output: <label>6</label>

