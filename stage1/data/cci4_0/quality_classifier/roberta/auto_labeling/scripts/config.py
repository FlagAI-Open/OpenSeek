EN_PROMPT_TEMPLATE_ADD = '''Below is an extract from a web page. Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching from primary school to college levels using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.
- Add another point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing style.
- Award a third point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a basic tutorial that is suitable for learning but has notable limitations like treating concepts that are too complex for grade school students. 
- Grant a fourth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren't too advanced for grade school students. The content is coherent, focused, and valuable for structured learning.
- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or college. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.

The extract:
<{EXAMPLE}>.

After examining the extract: 
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Educational score:  <total points>"'''

CN_PROMPT_TEMPLATE_ADD = '''以下是从网页中提取的内容。请评估该内容是否具有较高的教育价值，以及是否适用于从小学到大学各阶段的教学场景。评估时请使用以下的加分制评分系统（总分为5分），评分依据如下：

- 如果提取内容提供了一些与教育主题相关的基本信息，即使包括一些无关或非学术性内容（如广告和推广材料），加1分。
- 如果提取内容涉及某些与教育相关的元素，但未能紧密符合教育标准，可能将教育内容与非教育材料混杂在一起，提供了潜在有用主题的粗略概述，或以不连贯的写作风格呈现信息，再加1分。
- 如果提取内容适合教育用途，且介绍了与学校课程相关的关键概念，即使内容不够全面或包括一些无关信息，加第3分。这部分内容可能类似于教材的引言部分或基础教程，适合学习但存在明显限制，如概念过于复杂而不适合小学阶段学生。
- 如果提取内容高度相关且对教育目的非常有帮助，适合不高于小学水平的学习，加第4分。内容类似于教材章节或教程，提供了大量教育内容，包括练习和解答，几乎没有无关信息，且概念不会超出小学阶段学生的理解范围。内容清晰、集中，对结构化学习具有较大价值。
- 如果提取内容的教育价值极高，完全适用于小学或大学教学，加第5分。这类内容具有详细的推理过程，写作风格易于理解，并对主题提供了深刻且全面的见解，无任何非教育性或复杂内容。

提取内容：  
<{EXAMPLE}>。

评估完成后：  
- 简要说明评分依据（不超过100字）。  
- 使用以下格式给出评分：“教育得分：<总分>”。'''

CN_PROMPT_TEMPLATE_DIRECT = '''以下是一段网页内容摘录。请使用以下5分制评分系统来评估该网页的写作水平、教育价值和实用性:
0分：如果网页没有提供任何教育价值,完全由无关信息(如广告、宣传材料、少儿不宜内容)组成。
1分：如果网页提供了一些可能有教育价值的基本信息,但包含较多的无关或非学术内容(如广告和宣传材料)。
2分：如果网页涉及某些与教育相关的元素,但与教育标准不太吻合。它可能将教育内容与非教育材料混杂,对潜在的有用的主题进行浅显概述,或以不连贯的写作风格呈现信息。
3分：如果网页适合教育使用,并介绍了与某些学校课程中可能学到的关键概念，或对个人发展有用的实用信息。它的内容连贯但可能不全面,或包含一些无关信息。它可能类似于教科书的一小段节选,可以学习但有明显局限,如涉及过于复杂的概念、过于具体的不重要事件。
4分：如果网页与教育高度相关，对个人学习发展有益,表现出清晰一致的写作风格。它可能类似于教科书的一个章节或教程,提供大量教育内容,极少包含无关信息,且概念对学生来说不会过于深奥。内容连贯、重点突出,对结构化学习有价值。
5分：如果网页摘录在教育价值上表现极好,完全适合小学、中学或大学教学或专业人士学习。它遵循详细的推理过程,写作风格易于理解,对主题提供深刻而全面的见解,不包含任何非教育性或无实用意义内容。

网页内容摘录:
{}

在审查这段网页摘录后：请简要地为您的评分进行合理的解释，最多不超过100字，最后以“教育得分：<分数>”的格式结束。请根据所列出的标准系统地赋予分数。'''

EN_PROMPT_TEMPLATE_DIRECT = '''Below is an extract from a webpage. Please evaluate its writing quality, educational value, and practical utility using the following 5-point scoring system:

**0 points:** If the webpage provides no educational value and is entirely composed of irrelevant content, such as advertisements, promotional material, or inappropriate content.  
**1 point:** If the webpage provides some basic information with potential educational value but includes a significant amount of irrelevant or non-academic content, such as ads or promotional material.  
**2 points:** If the webpage contains elements related to education but does not align well with educational standards. It may mix educational and non-educational content, offer a superficial overview of potentially useful topics, or present information in a disorganized and incoherent writing style.  
**3 points:** If the webpage is suitable for educational purposes, introducing key concepts related to school curricula or providing practical information useful for personal development. The content is coherent but may lack comprehensiveness or include some irrelevant information. It might resemble a short textbook excerpt, suitable for learning but with notable limitations, such as overly complex concepts or trivial details.  
**4 points:** If the webpage is highly relevant to education and beneficial for personal learning or development, with a clear and consistent writing style. It may resemble a chapter of a textbook or a tutorial, offering substantial educational content with minimal irrelevant information. The concepts should be accessible to students and not overly advanced. The content is coherent, focused, and valuable for structured learning.  
**5 points:** If the webpage extract demonstrates exceptional educational value, making it fully suitable for teaching at elementary, secondary, or university levels, or for professional learning. It adheres to detailed reasoning, has an easy-to-understand writing style, and provides deep and comprehensive insights into the subject, without any non-educational or trivial content.

**Webpage extract:**  
{}

After reviewing the webpage extract:  
- Briefly justify your score in no more than 100 words.  
- Conclude with your score using the format: “Educational score: <score>”.  
Please assign the score systematically based on the criteria provided.'''



from openai import OpenAI

prompts = {
    "en-add": EN_PROMPT_TEMPLATE_ADD,
    "en-direct": EN_PROMPT_TEMPLATE_DIRECT,
    "cn-add": CN_PROMPT_TEMPLATE_ADD,
    "cn-direct": CN_PROMPT_TEMPLATE_DIRECT,
}

patterns = {
    "en-add": r"(?i)Educational score:\**\s*\**(\d+)",
    "en-direct": r"(?i)Educational score:\**\s*\**(\d+)",
    "cn-add": r"教育得分：\**\s*\**(\d+)",
    "cn-direct": r"教育得分：\**\s*\**(\d+)",    
}

models = {
    "Qwen2.5-72B-Instruct": OpenAI(
        api_key="EMPTY", base_url="your-local-url"
    ),
    "deepseek-chat": OpenAI(
        api_key="your-api-key", base_url="your-api-platform-url"
    ),
    "gpt-4o-2024-11-20": OpenAI(
        api_key="your-api-key", base_url="your-api-platform-url"
    ),
}

