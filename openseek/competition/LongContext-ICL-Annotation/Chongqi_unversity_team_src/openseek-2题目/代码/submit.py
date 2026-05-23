import json
import os
import time
import re
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import ast


def task2_data_loader(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return (
        data.get("task_id"),
        data.get("task_name"),
        data.get("Definition", []),
        data.get("examples", []),
        data.get("test_samples", [])
    )


client = OpenAI(
    api_key="dummy",
    base_url="https://flagos.io/flagos-lab/hw/node/HW-gpu57/port/22653/v1"
)


def qwen_api(messages, model="/Qwen3-4B/Qwen/Qwen3-4B", retries=3):
    for attempt in range(retries):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
            )
            return res.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1:
                print(f"\nAPI 调用失败: {e}")
                return ""
            time.sleep(5)

def parse_json_output(raw_output: str):
    raw_output = (raw_output or "").strip()

    if not raw_output:
        return {"reason": "", "output": []}

    # 先尝试直接按 JSON 解析
    try:
        data = json.loads(raw_output)
        if isinstance(data, dict):
            reason = str(data.get("reason", "")).strip()
            output = data.get("output", [])
            if isinstance(output, list):
                cleaned = [str(x).strip() for x in output if str(x).strip()]
                return {"reason": reason, "output": cleaned}
    except Exception:
        pass

    # 再尝试从文本中抓取 JSON 块
    match = re.search(r'\{.*\}', raw_output, flags=re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                reason = str(data.get("reason", "")).strip()
                output = data.get("output", [])
                if isinstance(output, list):
                    cleaned = [str(x).strip() for x in output if str(x).strip()]
                    return {"reason": reason, "output": cleaned}
        except Exception:
            pass

    # 最后兜底：兼容模型偶尔只返回列表
    try:
        items = ast.literal_eval(raw_output)
        if isinstance(items, list):
            cleaned = [str(x).strip() for x in items if str(x).strip()]
            return {"reason": "", "output": cleaned}
    except Exception:
        pass

    return {"reason": "", "output": []}


def parse_output_items(raw_output: str):
    parsed = parse_json_output(raw_output)
    return parsed["output"], parsed["reason"]


NOUN_SYSTEM_PROMPT = r"""
You are a dataset-aligned noun extractor.

Your goal is to EXACTLY match the dataset's noun-counting behavior,
NOT standard grammar.

The input always asks:
"Count the number of nouns in this sentence."

You must first analyze EVERY word in the sentence one by one based on its CONTEXT,
then return the noun units that the dataset would count.

==================================================
OUTPUT FORMAT
==================================================

Output ONLY a JSON object with exactly these two fields:

{"reason":"逐词分析过程","output":["word1","word2"]}

Requirements:
1. Output ONLY valid JSON
2. Must contain keys "reason" and "output"
3. "reason" must be a detailed Chinese string
4. "reason" MUST analyze each word one by one in sentence order
5. "output" must be a JSON array of strings
6. No markdown
7. No explanation outside JSON

==================================================
HOW TO WRITE "reason"
==================================================

The "reason" field MUST contain per-word analysis.

You MUST:
- analyze each word in sentence order
- explicitly state the contextual part of speech of each word
- explicitly state whether it is counted into output
- explain why

Use this style inside "reason":
1. word -> 在句中词性 / 是否计入 / 原因
2. word -> 在句中词性 / 是否计入 / 原因
3. word -> 在句中词性 / 是否计入 / 原因

Example format:
"1. A->冠词，不计入；2. man->名词，表示可见人物，计入；3. holding->动词，现在分词作动作，不按名词计入；4. a->冠词，不计入；5. bat->名词，表示可见物体，计入"

The "reason" must NOT be short or vague.
The "reason" must show the actual decision process for each word.

==================================================
TASK-SPECIFIC DEFINITION OF NOUN
==================================================

In this dataset, a noun is:
a visible entity, object, person, animal, place, scene-part,
or countable noun-like unit that appears in the caption.

Count nouns the way an image annotator would count visible things,
NOT the way a grammar textbook defines all nouns.

Main noun types:
- people: man, woman, boy, girl, child, people, person, player
- animals: dog, horse, elephant, bird, giraffe, zebra
- objects: bat, phone, chair, plate, suitcase, umbrella
- places / scene parts: room, street, court, kitchen, beach, field, wall, corner, window

==================================================
WHAT IS NOT A NOUN IN THIS TASK
==================================================

Do NOT count:
1. pronouns / placeholders:
it, he, she, they, them, this, that, something, anything, other

2. abstract or non-visual words:
idea, purpose, appearance, memory, life

3. meta image-description words:
image, picture, photo, photograph, view, scene, background, foreground

4. pure adjectives / colors / states:
small, large, red, white, black, green, empty, busy, open, closed, full

5. pure directions / positions when not used as concrete noun units:
front, back, side, top, bottom, middle

6. verb words used as actions instead of noun units:
walking, standing, sitting, holding, riding, grazing, playing

==================================================
CONTEXT RULE
==================================================

A word may have multiple parts of speech.
You MUST judge it from the sentence context, not from the word alone.

Examples:
- "jump" in "doing a jump" is a noun, not a verb
- "living" in "living room" is part of a noun compound
- "holding" in "a man holding a bat" is not a noun

==================================================
COMPOUND NOUN RULE
==================================================

Often split:
- tennis court -> ["tennis", "court"]
- tennis player -> ["tennis", "player"]
- baseball bat -> ["baseball", "bat"]
- baseball game -> ["baseball", "game"]
- cell phone -> ["cell", "phone"]
- living room -> ["living", "room"]
- dining room -> ["dining", "room"]
- hotel room -> ["hotel", "room"]
- parking lot -> ["parking", "lot"]
- street sign -> ["street", "sign"]
- clock tower -> ["clock", "tower"]
- fire hydrant -> ["fire", "hydrant"]
- teddy bear -> ["teddy", "bear"]
- hot dog -> ["hot", "dog"]
- peanut butter -> ["peanut", "butter"]
- video game -> ["video", "game"]

Usually not split:
- motor bike -> ["bike"]
- ski slope -> ["slope"]
- swiss army knife -> ["knife"]
- soft drink -> ["drink"]

==================================================
"OF" STRUCTURE RULE
==================================================

Often count BOTH parts:
- group of people -> ["group", "people"]
- bunch of bananas -> ["bunch", "bananas"]
- couple of cars -> ["couple", "cars"]
- pair of zebras -> ["pair", "zebras"]
- pile of food -> ["pile", "food"]
- piece of cake -> ["piece", "cake"]
- slice of pizza -> ["slice", "pizza"]
- lot of flowers -> ["lot", "flowers"]

==================================================
FEW-SHOT EXAMPLES
==================================================

Sentence: 'A man that has a baseball bat in the dirt'
Output:

{"reason":"1. A->冠词，不计入；2. man->名词，表示可见人物，计入；3. that->关系词，不计入；4. has->动词，表示拥有，不按名词计入；5. a->冠词，不计入；6. baseball->名词，在 baseball bat 中按数据集作为复合名词前项计入；7. bat->名词，表示可见物体，计入；8. in->介词，不计入；9. the->冠词，不计入；10. dirt->名词，表示可见场景物，计入","output":["man","baseball","bat","dirt"]}

Sentence: 'A woman talking on a cell phone walking down a street'
Output:

{"reason":"1. A->冠词，不计入；2. woman->名词，表示可见人物，计入；3. talking->动词，现在分词表示动作，不按名词计入；4. on->介词，不计入；5. a->冠词，不计入；6. cell->名词，在 cell phone 中按数据集作为复合名词前项计入；7. phone->名词，表示可见物体，计入；8. walking->动词，现在分词表示动作，不按名词计入；9. down->副词/方向成分，不计入；10. a->冠词，不计入；11. street->名词，表示场景地点，计入","output":["woman","cell","phone","street"]}

Sentence: 'a living room with some brick walls and a fireplace'
Output:

{"reason":"1. a->冠词，不计入；2. living->名词性成分，在 living room 中按数据集计入；3. room->名词，表示场景地点，计入；4. with->介词，不计入；5. some->限定词，不计入；6. brick->修饰成分，修饰 walls，不单独计入；7. walls->名词，表示可见场景部分，计入；8. and->连词，不计入；9. a->冠词，不计入；10. fireplace->名词，表示可见物体，计入","output":["living","room","walls","fireplace"]}

Sentence: 'A person holding an umbrella in front of a building'
Output:

{"reason":"1. A->冠词，不计入；2. person->名词，表示可见人物，计入；3. holding->动词，现在分词表示动作，不按名词计入；4. an->冠词，不计入；5. umbrella->名词，表示可见物体，计入；6. in->介词，不计入；7. front->方位词，在 in front of 结构中不按名词计入；8. of->介词，不计入；9. a->冠词，不计入；10. building->名词，表示可见建筑，计入","output":["person","umbrella","building"]}

Sentence: 'A little girl holding a teddy bear'
Output:
{"reason":"1. A->冠词，不计入；2. little->形容词，修饰 girl，不计入；3. girl->名词，表示可见人物，计入；4. holding->动词，现在分词表示动作，不按名词计入；5. a->冠词，不计入；6. teddy->名词，在 teddy bear 中按数据集作为复合名词前项计入；7. bear->名词，表示可见物体，计入","output":["girl","teddy","bear"]}

Sentence: 'Ironic picture of man and woman walking up a sidewalk under a "Wrong Way" sign'
Output:

{"reason":"1. Ironic->形容词，不计入；2. picture->名词，表示可见物体，计入；3. of->介词，不计入；4. man->名词，表示可见人物，计入；5. and->连词，不计入；6. woman->名词，表示可见人物，计入；7. walking->动词，现在分词表示动作，不按名词计入；8. up->副词/方向成分，不计入；9. a->冠词，不计入；10. sidewalk->名词，表示可见场景，计入；11. under->介词，不计入；12. a->冠词，不计入；13. Wrong->形容词，修饰 sign，不计入；14. Way->名词，在 Wrong Way 中按复合名词前项计入；15. sign->名词，表示可见物体，计入","output":["picture","man","woman","sidewalk","Way","sign"]}

Sentence: 'a gentleman in pajamas taking a selfie with his camera'
Output:

{"reason":"1. a->冠词，不计入；2. gentleman->名词，表示可见人物，计入；3. in->介词，不计入；4. pajamas->名词，表示衣物，计入；5. taking->动词，现在分词表示动作，不计入；6. a->冠词，不计入；7. selfie->名词，表示可见物体，计入；8. with->介词，不计入；9. his->限定词，不计入；10. camera->名词，表示可见物体，计入","output":["gentleman","pajamas","selfie","camera"]}

Sentence: 'A little girl with an broken arm posing near a restroom sink and toilet'
Output:

{"reason":"1. A->冠词，不计入；2. little->形容词，不计入；3. girl->名词，表示可见人物，计入；4. with->介词，不计入；5. an->冠词，不计入；6. broken->形容词，不计入；7. arm->名词，表示身体部位，计入；8. posing->动词，现在分词表示动作，不计入；9. near->介词，不计入；10. a->冠词，不计入；11. restroom->名词，表示场景地点，计入；12. sink->名词，表示可见物体，计入；13. and->连词，不计入；14. toilet->名词，表示可见物体，计入","output":["girl","arm","restroom","sink","toilet"]}

Sentence: 'a couple of soldiers putting cheese on a pizza'
Output:

{"reason":"1. a->冠词，不计入；2. couple->名词，表示数量概念，可计入；3. of->介词，不计入；4. soldiers->名词，表示可见人物，计入；5. putting->动词，现在分词表示动作，不计入；6. cheese->名词，表示可见物体，计入；7. on->介词，不计入；8. a->冠词，不计入；9. pizza->名词，表示可见物体，计入","output":["couple","soldiers","cheese","pizza"]}

Sentence: 'A tennis player holding a racket on the tennis court'
Output:

{"reason":"1. A->冠词，不计入；2. tennis->名词性成分，在 tennis player 中按复合名词前项计入；3. player->名词，表示可见人物，计入；4. holding->动词，现在分词表示动作，不计入；5. a->冠词，不计入；6. racket->名词，表示可见物体，计入；7. on->介词，不计入；8. the->冠词，不计入；9. tennis->名词性成分，在 tennis court 中按复合名词前项计入；10. court->名词，表示场景地点，计入","output":["tennis","player","racket","tennis","court"]}

Sentence: 'A home with no furniture and a dog in it'
Output:

{"reason":"1. A->冠词，不计入；2. home->名词，表示场景地点，计入；3. with->介词，不计入；4. no->限定词，不计入；5. furniture->名词，表示可见物体，计入；6. and->连词，不计入；7. a->冠词，不计入；8. dog->名词，表示可见动物，计入；9. in->介词，不计入；10. it->代词，不计入","output":["home","furniture","dog"]}

Sentence: 'Two people out skiing on the ski slope'
Output:

{"reason":"1. Two->限定词，不计入；2. people->名词，表示可见人物，计入；3. out->副词，不计入；4. skiing->动词，现在分词表示动作，不计入；5. on->介词，不计入；6. the->冠词，不计入；7. ski->名词性成分，在 ski slope 中按复合名词前项计入；8. slope->名词，表示场景地点，计入","output":["people","ski","slope"]}

Sentence: 'Several men riding in a canoe across the water'
Output:

{"reason":"1. Several->限定词，不计入；2. men->名词，表示可见人物，计入；3. riding->动词，现在分词表示动作，不计入；4. in->介词，不计入；5. a->冠词，不计入；6. canoe->名词，表示可见物体，计入；7. across->介词/副词，不计入；8. the->冠词，不计入；9. water->名词，表示场景水体，计入","output":["men","canoe","water"]}

Sentence: 'The hotdog has mustard and bacon on it'
Output:

{"reason":"1. The->冠词，不计入；2. hotdog->名词，表示可见物体，计入；3. has->动词，不计入；4. mustard->名词，表示可见物体，计入；5. and->连词，不计入；6. bacon->名词，表示可见物体，计入；7. on->介词，不计入；8. it->代词，不计入","output":["hotdog","mustard","bacon"]}

Sentence: 'Two sinks and some cupboards in a bathroom'
Output:

{"reason":"1. Two->限定词，不计入；2. sinks->名词，表示可见物体，计入；3. and->连词，不计入；4. some->限定词，不计入；5. cupboards->名词，表示可见物体，计入；6. in->介词，不计入；7. a->冠词，不计入；8. bathroom->名词，表示场景地点，计入","output":["sinks","cupboards","bathroom"]}

Sentence: 'An old historical clock with arched design nearby'
Output:

{"reason":"1. An->冠词，不计入；2. old->形容词，不计入；3. historical->形容词，不计入；4. clock->名词，表示可见物体，计入；5. with->介词，不计入；6. arched->形容词，不计入；7. design->名词，表示可见物体，计入；8. nearby->副词，不计入","output":["clock","design"]}

Sentence: 'A girl is holding a paper up over her face as a man is shown behind in a mirror talking on a phone'
Output:

{"reason":"1. A->冠词，不计入；2. girl->名词，表示可见人物，计入；3. is->动词，不计入；4. holding->动词，现在分词表示动作，不计入；5. a->冠词，不计入；6. paper->名词，表示可见物体，计入；7. up->副词，不计入；8. over->介词，不计入；9. her->限定词，不计入；10. face->名词，表示身体部位，计入；11. as->连词，不计入；12. a->冠词，不计入；13. man->名词，表示可见人物，计入；14. is->动词，不计入；15. shown->动词，不计入；16. behind->副词/方向，不计入；17. in->介词，不计入；18. a->冠词，不计入；19. mirror->名词，表示可见物体，计入；20. talking->动词，现在分词，不计入；21. on->介词，不计入；22. a->冠词，不计入；23. phone->名词，表示可见物体，计入","output":["girl","paper","face","man","mirror","phone"]}

Sentence: 'A small group of sheep graze in a mountain area'
Output:

{"reason":"1. A->冠词，不计入；2. small->形容词，不计入；3. group->名词，表示集合概念，计入；4. of->介词，不计入；5. sheep->名词，表示可见动物，计入；6. graze->动词，不计入；7. in->介词，不计入；8. a->冠词，不计入；9. mountain->名词，表示场景地点，计入；10. area->名词，表示场景地点，计入","output":["group","sheep","mountain","area"]}

{"reason":"1. A->冠词，不计入；2. small->形容词，不计入；3. toilet->名词，表示可见物体，计入；4. sits->动词，不计入；5. in->介词，不计入；6. the->冠词，不计入；7. corner->名词，表示场景部分，计入；8. of->介词，不计入；9. a->冠词，不计入；10. bare->形容词，不计入；11. room->名词，表示场景地点，计入","output":["toilet","corner","room"]}

Sentence: 'The view from inside a kitchen to outside a window at dusk'
Output:

{"reason":"1. The->冠词，不计入；2. view->名词，表示可见物体/场景视角，计入；3. from->介词，不计入；4. inside->介词/方向，不计入；5. a->冠词，不计入；6. kitchen->名词，表示场景地点，计入；7. to->介词，不计入；8. outside->副词/方向，不计入；9. a->冠词，不计入；10. window->名词，表示可见物体，计入；11. at->介词，不计入；12. dusk->名词，表示时间名词，计入","output":["view","kitchen","window","dusk"]}

Sentence: 'A man that has a baseball bat in the dirt'
Output:

{"reason":"1. A->冠词，不计入；2. man->名词，表示可见人物，计入；3. that->关系词，不计入；4. has->动词，不计入；5. a->冠词，不计入；6. baseball->名词，在 baseball bat 中按复合名词前项计入；7. bat->名词，表示可见物体，计入；8. in->介词，不计入；9. the->冠词，不计入；10. dirt->名词，表示可见场景物，计入","output":["man","baseball","bat","dirt"]}

Sentence: 'A group of people gather together at a street corner'
Output:

{"reason":"1. A->冠词，不计入；2. group->名词，表示集合概念，计入；3. of->介词，不计入；4. people->名词，表示可见人物，计入；5. gather->动词，不计入；6. together->副词，不计入；7. at->介词，不计入；8. a->冠词，不计入；9. street->名词，表示场景地点，计入；10. corner->名词，表示场景部分，计入","output":["group","people","street","corner"]}

Sentence: 'A man is working on a multi color airplane'
Output:

{"reason":"1. A->冠词，不计入；2. man->名词，表示可见人物，计入；3. is->动词，不计入；4. working->动词，现在分词表示动作，不计入；5. on->介词，不计入；6. a->冠词，不计入；7. multi->形容词，不计入；8. color->形容词修饰 airplane，不单独计入；9. airplane->名词，表示可见物体，计入","output":["man","airplane"]}

Sentence: 'Several people looking at books and magazines at an outdoor zine library'
Output:

{"reason":"1. Several->限定词，不计入；2. people->名词，表示可见人物，计入；3. looking->动词，现在分词表示动作，不计入；4. at->介词，不计入；5. books->名词，表示可见物体，计入；6. and->连词，不计入；7. magazines->名词，表示可见物体，计入；8. at->介词，不计入；9. an->冠词，不计入；10. outdoor->形容词，不计入；11. zine->名词，表示可见物体/出版物，计入；12. library->名词，表示场景地点，计入","output":["people","books","magazines","zine","library"]}

Sentence: 'An old brick building contains an appliance store'
Output:

{"reason":"1. An->冠词，不计入；2. old->形容词，不计入；3. brick->形容词/修饰 building，不单独计入；4. building->名词，表示场景建筑，计入；5. contains->动词，不计入；6. an->冠词，不计入；7. appliance->名词，在 appliance store 中按复合名词前项计入；8. store->名词，表示可见物体/商铺，计入","output":["building","appliance","store"]}

Sentence: 'A scooter riding down the road, next to a building'
Output:

{"reason":"1. A->冠词，不计入；2. scooter->名词，表示可见物体，计入；3. riding->动词，现在分词表示动作，不计入；4. down->副词，不计入；5. the->冠词，不计入；6. road->名词，表示场景道路，计入；7. next->副词/方向，不计入；8. to->介词，不计入；9. a->冠词，不计入；10. building->名词，表示场景建筑，计入","output":["scooter","road","building"]}

Sentence: 'A rusty green truck is parked among some weeds'
Output:

{"reason":"1. A->冠词，不计入；2. rusty->形容词，不计入；3. green->形容词，不计入；4. truck->名词，表示可见物体，计入；5. is->动词，不计入；6. parked->动词，不计入；7. among->介词，不计入；8. some->限定词，不计入；9. weeds->名词，表示可见植物，计入","output":["truck","weeds"]}

Sentence: 'A white kitchen with a large refrigerator freezer combo'
Output:

{"reason":"1. A->冠词，不计入；2. white->形容词，不计入；3. kitchen->名词，表示场景地点，计入；4. with->介词，不计入；5. a->冠词，不计入；6. large->形容词，不计入；7. refrigerator->名词，表示可见物体，计入；8. freezer->名词，表示可见物体，计入；9. combo->名词，表示可见物体，计入","output":["kitchen","refrigerator","freezer","combo"]}

Sentence: 'The display has many towers of stacked cookies next to trays full of cookies'
Output:

{"reason":"1. The->冠词，不计入；2. display->名词，表示可见物体，计入；3. has->动词，不计入；4. many->限定词，不计入；5. towers->名词，表示可见物体/堆，计入；6. of->介词，不计入；7. stacked->形容词，不计入；8. cookies->名词，表示可见物体，计入；9. next->副词/方向，不计入；10. to->介词，不计入；11. trays->名词，表示可见物体，计入；12. full->形容词，不计入；13. of->介词，不计入；14. cookies->名词，表示可见物体，计入","output":["display","towers","cookies","trays","cookies"]}

Sentence: 'Artificial lowers line the dashboard of a car in a busy area'
Output:

{"reason":"1. Artificial->形容词，不计入；2. lowers->名词，表示可见物体，计入；3. line->动词，不计入；4. the->冠词，不计入；5. dashboard->名词，表示可见物体，计入；6. of->介词，不计入；7. a->冠词，不计入；8. car->名词，表示可见物体/场景，计入；9. in->介词，不计入；10. a->冠词，不计入；11. busy->形容词，不计入；12. area->名词，表示场景地点，计入","output":["lowers","dashboard","car","area"]}

Sentence: 'A group of people gather together at a street corner'
Output:

{"reason":"1. A->冠词，不计入；2. group->名词，表示集合概念，计入；3. of->介词，不计入；4. people->名词，表示可见人物，计入；5. gather->动词，不计入；6. together->副词，不计入；7. at->介词，不计入；8. a->冠词，不计入；9. street->名词，表示场景地点，计入；10. corner->名词，表示场景部分，计入","output":["group","people","street","corner"]}

Sentence: 'A man is working on a multi color airplane'
Output:

{"reason":"1. A->冠词，不计入；2. man->名词，表示可见人物，计入；3. is->动词，不计入；4. working->动词，现在分词表示动作，不计入；5. on->介词，不计入；6. a->冠词，不计入；7. multi->形容词，不计入；8. color->形容词修饰 airplane，不单独计入；9. airplane->名词，表示可见物体，计入","output":["man","airplane"]}

Sentence: 'Several people looking at books and magazines at an outdoor zine library'
Output:

{"reason":"1. Several->限定词，不计入；2. people->名词，表示可见人物，计入；3. looking->动词，现在分词表示动作，不计入；4. at->介词，不计入；5. books->名词，表示可见物体，计入；6. and->连词，不计入；7. magazines->名词，表示可见物体，计入；8. at->介词，不计入；9. an->冠词，不计入；10. outdoor->形容词，不计入；11. zine->名词，表示可见物体/出版物，计入；12. library->名词，表示场景地点，计入","output":["people","books","magazines","zine","library"]}

Sentence: 'An old brick building contains an appliance store'
Output:

{"reason":"1. An->冠词，不计入；2. old->形容词，不计入；3. brick->形容词/修饰 building，不单独计入；4. building->名词，表示场景建筑，计入；5. contains->动词，不计入；6. an->冠词，不计入；7. appliance->名词，在 appliance store 中按复合名词前项计入；8. store->名词，表示可见物体/商铺，计入","output":["building","appliance","store"]}

Sentence: 'A scooter riding down the road, next to a building'
Output:

{"reason":"1. A->冠词，不计入；2. scooter->名词，表示可见物体，计入；3. riding->动词，现在分词表示动作，不计入；4. down->副词，不计入；5. the->冠词，不计入；6. road->名词，表示场景道路，计入；7. next->副词/方向，不计入；8. to->介词，不计入；9. a->冠词，不计入；10. building->名词，表示场景建筑，计入","output":["scooter","road","building"]}

Sentence: 'A rusty green truck is parked among some weeds'
Output:

{"reason":"1. A->冠词，不计入；2. rusty->形容词，不计入；3. green->形容词，不计入；4. truck->名词，表示可见物体，计入；5. is->动词，不计入；6. parked->动词，不计入；7. among->介词，不计入；8. some->限定词，不计入；9. weeds->名词，表示可见植物，计入","output":["truck","weeds"]}

Sentence: 'A white kitchen with a large refrigerator freezer combo'
Output:

{"reason":"1. A->冠词，不计入；2. white->形容词，不计入；3. kitchen->名词，表示场景地点，计入；4. with->介词，不计入；5. a->冠词，不计入；6. large->形容词，不计入；7. refrigerator->名词，表示可见物体，计入；8. freezer->名词，表示可见物体，计入；9. combo->名词，表示可见物体，计入","output":["kitchen","refrigerator","freezer","combo"]}

Sentence: 'The display has many towers of stacked cookies next to trays full of cookies'
Output:

{"reason":"1. The->冠词，不计入；2. display->名词，表示可见物体，计入；3. has->动词，不计入；4. many->限定词，不计入；5. towers->名词，表示可见物体/堆，计入；6. of->介词，不计入；7. stacked->形容词，不计入；8. cookies->名词，表示可见物体，计入；9. next->副词/方向，不计入；10. to->介词，不计入；11. trays->名词，表示可见物体，计入；12. full->形容词，不计入；13. of->介词，不计入；14. cookies->名词，表示可见物体，计入","output":["display","towers","cookies","trays","cookies"]}

Sentence: 'Artificial lowers line the dashboard of a car in a busy area'
Output:

{"reason":"1. Artificial->形容词，不计入；2. lowers->名词，表示可见物体，计入；3. line->动词，不计入；4. the->冠词，不计入；5. dashboard->名词，表示可见物体，计入；6. of->介词，不计入；7. a->冠词，不计入；8. car->名词，表示可见物体/场景，计入；9. in->介词，不计入；10. a->冠词，不计入；11. busy->形容词，不计入；12. area->名词，表示场景地点，计入","output":["lowers","dashboard","car","area"]}

Sentence: 'A group of people wave while riding a ski lift'
Output:

{"reason":"1. A->冠词，不计入；2. group->名词，表示集合/群体，计入；3. of->介词，不计入；4. people->名词，表示人，计入；5. wave->动词，不计入；6. while->连词，不计入；7. riding->动词，不计入；8. a->冠词，不计入；9. ski->名词作定语修饰lift，但在某些标注体系中可能被视为名词部分，这里根据示例逻辑，ski lift整体视为一个物体或分别计数。参考示例1输出为3，通常指group, people, ski lift(或ski, lift)。若按严格独立名词：group, people, lift。若ski视为名词修饰语：group, people, ski, lift (4个)。但示例给的是3。让我们看其他例子。示例2: elephants, area (2). 示例3: league, player, bat, game (4? 示例给5: little, league, player, bat, game? 不对，little是形容词。可能是a, little, league, player, holding, a, bat, during, part, of, a, game. Nouns: league, player, bat, part, game = 5. 所以little不算。回到本句：group, people, ski lift. 如果ski lift算两个：group, people, ski, lift = 4. 如果算一个：group, people, ski lift = 3. 示例输出3，故认定ski lift为一个复合名词单位或仅计算核心名词lift而忽略ski的独立名词属性，或者group不算？不，group肯定是名词。最可能的解释是：group, people, lift (ski作为形容词性用法) -> 3个。","output":["group","people","lift"]}

Sentence: 'Some very cute elephants in a grassy area'
Output:

{"reason":"1. Some->限定词，不计入；2. very->副词，不计入；3. cute->形容词，不计入；4. elephants->名词，表示可见物体/人，计入；5. in->介词，不计入；6. a->冠词，不计入；7. grassy->形容词，不计入；8. area->名词，表示场景地点，计入","output":["elephants","area"]}

Sentence: 'a little league plater holding a bat during part of a game'
Output:

{"reason":"1. a->冠词，不计入；2. little->形容词，不计入；3. league->名词，表示组织/类别，计入；4. plater->名词（应为player），表示人，计入；5. holding->动词，不计入；6. a->冠词，不计入；7. bat->名词，表示可见物体，计入；8. during->介词，不计入；9. part->名词，表示部分/片段，计入；10. of->介词，不计入；11. a->冠词，不计入；12. game->名词，表示事件/活动，计入","output":["league","plater","bat","part","game"]}

Sentence: 'A brown table holding a vase and three flowers'
Output:

{"reason":"1. A->冠词，不计入；2. brown->形容词，不计入；3. table->名词，表示可见物体，计入；4. holding->动词，不计入；5. a->冠词，不计入；6. vase->名词，表示可见物体，计入；7. and->连词，不计入；8. three->数词，不计入；9. flowers->名词，表示可见物体，计入","output":["table","vase","flowers"]}

Sentence: 'A small child looking in a refrigerator with her bottom showing'
Output:

{"reason":"1. A->冠词，不计入；2. small->形容词，不计入；3. child->名词，表示人，计入；4. looking->动词，不计入；5. in->介词，不计入；6. a->冠词，不计入；7. refrigerator->名词，表示可见物体，计入；8. with->介词，不计入；9. her->代词，不计入；10. bottom->名词，表示身体部位，计入；11. showing->动词，不计入","output":["child","refrigerator","bottom"]}

Sentence: 'A snow skier slows down while skiing on a slope'
Output:

{"reason":"1. A->冠词，不计入；2. snow->名词作定语修饰skier，通常不计入独立名词，或计入？参考示例1中ski lift的处理。这里skier是人。snow skier整体作为一个角色。如果snow不计入，则只有skier, slope。如果snow计入，则有3个。示例输出3。那么snow必须计入，或者slope和skier之外还有一个？down是副词。while连词。skiing动词。on介词。a冠词。slope名词。skier名词。snow名词。共3个：snow, skier, slope。","output":["snow","skier","slope"]}

Sentence: 'A man who is speaking at a podium'
Output:

{"reason":"1. A->冠词，不计入；2. man->名词，表示人，计入；3. who->关系代词，不计入；4. is->动词，不计入；5. speaking->动词，不计入；6. at->介词，不计入；7. a->冠词，不计入；8. podium->名词，表示可见物体，计入","output":["man","podium"]}

Sentence: 'Baby sits inside an empty suitcase on top of a bed'
Output:

{"reason":"1. Baby->名词，表示人，计入；2. sits->动词，不计入；3. inside->介词，不计入；4. an->冠词，不计入；5. empty->形容词，不计入；6. suitcase->名词，表示可见物体，计入；7. on->介词，不计入；8. top->名词，表示位置/部分，计入；9. of->介词，不计入；10. a->冠词，不计入；11. bed->名词，表示可见物体，计入","output":["Baby","suitcase","top","bed"]}

Sentence: 'The mountains sit in the background of this quaint town'
Output:

{"reason":"1. The->冠词，不计入；2. mountains->名词，表示场景/自然物体，计入；3. sit->动词，不计入；4. in->介词，不计入；5. the->冠词，不计入；6. background->名词，表示位置/概念，计入；7. of->介词，不计入；8. this->限定词，不计入；9. quaint->形容词，不计入；10. town->名词，表示场景地点，计入","output":["mountains","background","town"]}

Sentence: 'A diner with large pepsi signs on the front of it'
Output:

{"reason":"1. A->冠词，不计入；2. diner->名词，表示场所/人，计入；3. with->介词，不计入；4. large->形容词，不计入；5. pepsi->专有名词作定语修饰signs，计入；6. signs->名词，表示可见物体，计入；7. on->介词，不计入；8. the->冠词，不计入；9. front->名词，表示位置/部分，计入；10. of->介词，不计入；11. it->代词，不计入","output":["diner","pepsi","signs","front"]}

Sentence: 'There is a sign in a foreign language and a street light in this picture'
Output:

{"reason":"1. There->副词/引导词，不计入；2. is->动词，不计入；3. a->冠词，不计入；4. sign->名词，表示可见物体，计入；5. in->介词，不计入；6. a->冠词，不计入；7. foreign->形容词，不计入；8. language->名词，表示抽象概念/系统，计入；9. and->连词，不计入；10. a->冠词，不计入；11. street->名词作定语修饰light，计入；12. light->名词，表示可见物体，计入；13. in->介词，不计入；14. this->限定词，不计入；15. picture->名词，表示图像/场景，计入","output":["sign","language","street","light","picture"]}

Sentence: 'A toilet in a small room with a window and unfinished walls'
Output:

{"reason":"1. A->冠词，不计入；2. toilet->名词，表示可见物体，计入；3. in->介词，不计入；4. a->冠词，不计入；5. small->形容词，不计入；6. room->名词，表示场景地点，计入；7. with->介词，不计入；8. a->冠词，不计入；9. window->名词，表示可见物体/建筑部件，计入；10. and->连词，不计入；11. unfinished->形容词，不计入；12. walls->名词，表示可见物体/建筑部件，计入","output":["toilet","room","window","walls"]}

Sentence: 'A brown horse grazing in its fenced in pen'
Output:

{"reason":"1. A->冠词，不计入；2. brown->形容词，不计入；3. horse->名词，表示动物，计入；4. grazing->动词，不计入；5. in->介词，不计入；6. its->代词，不计入；7. fenced->形容词，不计入；8. in->介词，不计入；9. pen->名词，表示场景/围栏，计入","output":["horse","pen"]}

Sentence: 'Small girl looking inside decorated refrigerator and reaching for something inside'
Output:

{"reason":"1. Small->形容词，不计入；2. girl->名词，表示人，计入；3. looking->动词，不计入；4. inside->介词，不计入；5. decorated->形容词，不计入；6. refrigerator->名词，表示可见物体，计入；7. and->连词，不计入；8. reaching->动词，不计入；9. for->介词，不计入；10. something->代词，不计入；11. inside->副词/介词，不计入","output":["girl","refrigerator"]}

Sentence: 'A man riding skis while holding ski pole'
Output:

{"reason":"1. A->冠词，不计入；2. man->名词，表示人，计入；3. riding->动词，不计入；4. skis->名词，表示可见物体，计入；5. while->连词，不计入；6. holding->动词，不计入；7. ski->名词作定语修饰pole，计入；8. pole->名词，表示可见物体，计入","output":["man","skis","ski","pole"]}

Sentence: 'One black laptop and one white laptop sitting on a white clothed table'
Output:

{"reason":"1. One->数词，不计入；2. black->形容词，不计入；3. laptop->名词，表示可见物体，计入；4. and->连词，不计入；5. one->数词，不计入；6. white->形容词，不计入；7. laptop->名词，表示可见物体，计入；8. sitting->动词，不计入；9. on->介词，不计入；10. a->冠词，不计入；11. white->形容词，不计入；12. clothed->形容词，不计入；13. table->名词，表示可见物体，计入","output":["laptop","laptop","table"]}

Sentence: 'a person making food on a stove in a kichen'
Output:

{"reason":"1. a->冠词，不计入；2. person->名词，表示人，计入；3. making->动词，不计入；4. food->名词，表示可见物体，计入；5. on->介词，不计入；6. a->冠词，不计入；7. stove->名词，表示可见物体，计入；8. in->介词，不计入；9. a->冠词，不计入；10. kichen->名词（拼写错误，应为kitchen），表示场景地点，计入","output":["person","food","stove","kichen"]}

Sentence: 'A view of a beach with closed umbrellas at sunset'
Output:

{"reason":"1. A->冠词，不计入；2. view->名词，表示视觉内容/场景，计入；3. of->介词，不计入；4. a->冠词，不计入；5. beach->名词，表示场景地点，计入；6. with->介词，不计入；7. closed->形容词，不计入；8. umbrellas->名词，表示可见物体，计入；9. at->介词，不计入；10. sunset->名词，表示时间/自然现象，计入","output":["view","beach","umbrellas","sunset"]}

Sentence: 'A white plate topped with a pile of food'
Output:

{"reason":"1. A->冠词，不计入；2. white->形容词，不计入；3. plate->名词，表示可见物体，计入；4. topped->动词/分词，不计入；5. with->介词，不计入；6. a->冠词，不计入；7. pile->名词，表示形状/集合，计入；8. of->介词，不计入；9. food->名词，表示可见物体，计入","output":["plate","pile","food"]}

Sentence: 'Black and white photograph of a vase in display case'
Output:

{"reason":"1. Black->形容词，不计入；2. and->连词，不计入；3. white->形容词，不计入；4. photograph->名词，表示图像/物体，计入；5. of->介词，不计入；6. a->冠词，不计入；7. vase->名词，表示可见物体，计入；8. in->介词，不计入；9. display->名词作定语修饰case，计入；10. case->名词，表示可见物体，计入","output":["photograph","vase","display","case"]}

Sentence: 'A teddy bear sitting in a chair with an open book on its lap'
Output:

{"reason":"1. A->冠词，不计入；2. teddy->名词作定语修饰bear，计入；3. bear->名词，表示玩具/动物，计入；4. sitting->动词，不计入；5. in->介词，不计入；6. a->冠词，不计入；7. chair->名词，表示可见物体，计入；8. with->介词，不计入；9. an->冠词，不计入；10. open->形容词，不计入；11. book->名词，表示可见物体，计入；12. on->介词，不计入；13. its->代词，不计入；14. lap->名词，表示身体部位，计入","output":["teddy","bear","chair","book","lap"]}

Sentence: 'Two men are in a cart going down the beach'
Output:

{"reason":"1. Two->数词，不计入；2. men->名词，表示人，计入；3. are->动词，不计入；4. in->介词，不计入；5. a->冠词，不计入；6. cart->名词，表示可见物体，计入；7. going->动词，不计入；8. down->副词/介词，不计入；9. the->冠词，不计入；10. beach->名词，表示场景地点，计入","output":["men","cart","beach"]}

Sentence: 'A yellow motorcycle parked on the curb of a street'
Output:

{"reason":"1. A->冠词，不计入；2. yellow->形容词，不计入；3. motorcycle->名词，表示可见物体，计入；4. parked->动词/分词，不计入；5. on->介词，不计入；6. the->冠词，不计入；7. curb->名词，表示可见物体/建筑部件，计入；8. of->介词，不计入；9. a->冠词，不计入；10. street->名词，表示场景地点，计入","output":["motorcycle","curb","street"]}

Sentence: 'An antenna device that is strapped to the bottom of an aircraft, looking down on trees below'
Output:

{"reason":"1. An->冠词，不计入；2. antenna->名词作定语修饰device，计入；3. device->名词，表示可见物体，计入；4. that->关系代词，不计入；5. is->动词，不计入；6. strapped->动词，不计入；7. to->介词，不计入；8. the->冠词，不计入；9. bottom->名词，表示位置/部分，计入；10. of->介词，不计入；11. an->冠词，不计入；12. aircraft->名词，表示可见物体，计入；13. looking->动词，不计入；14. down->副词，不计入；15. on->介词，不计入；16. trees->名词，表示可见物体/植物，计入；17. below->副词，不计入","output":["antenna","device","bottom","aircraft","trees"]}

Sentence: 'a big crowd watching a man throwing a baseball'
Output:

{"reason":"1. a->冠词，不计入；2. big->形容词，不计入；3. crowd->名词，表示集合/群体，计入；4. watching->动词，不计入；5. a->冠词，不计入；6. man->名词，表示人，计入；7. throwing->动词，不计入；8. a->冠词，不计入；9. baseball->名词，表示可见物体/运动项目，计入","output":["crowd","man","baseball"]}

Sentence: 'Three men are riding a spotted elephant in the middle of the park'
Output:

{"reason":"1. Three->数词，不计入；2. men->名词，表示人，计入；3. are->动词，不计入；4. riding->动词，不计入；5. a->冠词，不计入；6. spotted->形容词，不计入；7. elephant->名词，表示动物，计入；8. in->介词，不计入；9. the->冠词，不计入；10. middle->名词，表示位置，计入；11. of->介词，不计入；12. the->冠词，不计入；13. park->名词，表示场景地点，计入","output":["men","elephant","middle","park"]}

Sentence: 'Open sandwich and a New Zealand soft drink'
Output:

{"reason":"1. Open->形容词，不计入；2. sandwich->名词，表示可见物体，计入；3. and->连词，不计入；4. a->冠词，不计入；5. New->专有名词作定语，计入；6. Zealand->专有名词作定语，计入；7. soft->形容词，不计入；8. drink->名词，表示可见物体，计入","output":["sandwich","New","Zealand","drink"]}

Sentence: 'A group of people gather together at a street corner'
Output:

{"reason":"1. A->冠词，不计入；2. group->名词，表示集合概念，计入；3. of->介词，不计入；4. people->名词，表示可见人物，计入；5. gather->动词，不计入；6. together->副词，不计入；7. at->介词，不计入；8. a->冠词，不计入；9. street->名词，表示场景地点，计入；10. corner->名词，表示场景部分，计入","output":["group","people","street","corner"]}

Sentence: 'A man is working on a multi color airplane'
Output:

{"reason":"1. A->冠词，不计入；2. man->名词，表示可见人物，计入；3. is->动词，不计入；4. working->动词，现在分词表示动作，不计入；5. on->介词，不计入；6. a->冠词，不计入；7. multi->形容词，不计入；8. color->形容词修饰 airplane，不单独计入；9. airplane->名词，表示可见物体，计入","output":["man","airplane"]}

Sentence: 'Several people looking at books and magazines at an outdoor zine library'
Output:

{"reason":"1. Several->限定词，不计入；2. people->名词，表示可见人物，计入；3. looking->动词，现在分词表示动作，不计入；4. at->介词，不计入；5. books->名词，表示可见物体，计入；6. and->连词，不计入；7. magazines->名词，表示可见物体，计入；8. at->介词，不计入；9. an->冠词，不计入；10. outdoor->形容词，不计入；11. zine->名词，表示可见物体/出版物，计入；12. library->名词，表示场景地点，计入","output":["people","books","magazines","zine","library"]}

Sentence: 'An old brick building contains an appliance store'
Output:

{"reason":"1. An->冠词，不计入；2. old->形容词，不计入；3. brick->形容词/修饰 building，不单独计入；4. building->名词，表示场景建筑，计入；5. contains->动词，不计入；6. an->冠词，不计入；7. appliance->名词，在 appliance store 中按复合名词前项计入；8. store->名词，表示可见物体/商铺，计入","output":["building","appliance","store"]}

Sentence: 'A scooter riding down the road, next to a building'
Output:

{"reason":"1. A->冠词，不计入；2. scooter->名词，表示可见物体，计入；3. riding->动词，现在分词表示动作，不计入；4. down->副词，不计入；5. the->冠词，不计入；6. road->名词，表示场景道路，计入；7. next->副词/方向，不计入；8. to->介词，不计入；9. a->冠词，不计入；10. building->名词，表示场景建筑，计入","output":["scooter","road","building"]}

Sentence: 'A rusty green truck is parked among some weeds'
Output:

{"reason":"1. A->冠词，不计入；2. rusty->形容词，不计入；3. green->形容词，不计入；4. truck->名词，表示可见物体，计入；5. is->动词，不计入；6. parked->动词，不计入；7. among->介词，不计入；8. some->限定词，不计入；9. weeds->名词，表示可见植物，计入","output":["truck","weeds"]}

Sentence: 'A white kitchen with a large refrigerator freezer combo'
Output:

{"reason":"1. A->冠词，不计入；2. white->形容词，不计入；3. kitchen->名词，表示场景地点，计入；4. with->介词，不计入；5. a->冠词，不计入；6. large->形容词，不计入；7. refrigerator->名词，表示可见物体，计入；8. freezer->名词，表示可见物体，计入；9. combo->名词，表示可见物体，计入","output":["kitchen","refrigerator","freezer","combo"]}

Sentence: 'The display has many towers of stacked cookies next to trays full of cookies'
Output:

{"reason":"1. The->冠词，不计入；2. display->名词，表示可见物体，计入；3. has->动词，不计入；4. many->限定词，不计入；5. towers->名词，表示可见物体/堆，计入；6. of->介词，不计入；7. stacked->形容词，不计入；8. cookies->名词，表示可见物体，计入；9. next->副词/方向，不计入；10. to->介词，不计入；11. trays->名词，表示可见物体，计入；12. full->形容词，不计入；13. of->介词，不计入；14. cookies->名词，表示可见物体，计入","output":["display","towers","cookies","trays","cookies"]}

Sentence: 'Artificial lowers line the dashboard of a car in a busy area'
Output:

{"reason":"1. Artificial->形容词，不计入；2. lowers->名词，表示可见物体，计入；3. line->动词，不计入；4. the->冠词，不计入；5. dashboard->名词，表示可见物体，计入；6. of->介词，不计入；7. a->冠词，不计入；8. car->名词，表示可见物体/场景，计入；9. in->介词，不计入；10. a->冠词，不计入；11. busy->形容词，不计入；12. area->名词，表示场景地点，计入","output":["lowers","dashboard","car","area"]}

==================================================
FINAL INSTRUCTION
==================================================
The number of analysis items in "reason" should match the number of words/tokens in the sentence as closely as possible.
You MUST analyze every word in order inside "reason".
Do NOT skip words.
Do NOT write a short summary.
Do NOT just say "根据上下文判断".
Show the actual per-word decision process.

Return ONLY:
{"reason":"逐词分析过程","output":["...","..."]}
"""




VERB_SYSTEM_PROMPT = r"""
You are a dataset-aligned verb extractor.

Your goal is to EXACTLY match the dataset's verb-counting behavior,
NOT standard grammar.

The input always asks:
"Count the number of verbs in this sentence."

You must first analyze EVERY word in the sentence one by one based on its CONTEXT,
then return the verb units that the dataset would count.

==================================================
OUTPUT FORMAT
==================================================

Output ONLY a JSON object with exactly these two fields:

{"reason":"逐词分析过程","output":["word1","word2"]}

Requirements:
1. Output ONLY valid JSON
2. Must contain keys "reason" and "output"
3. "reason" must be a detailed Chinese string
4. "reason" MUST analyze each word one by one in sentence order
5. "output" must be a JSON array of strings
6. No markdown
7. No explanation outside JSON

==================================================
HOW TO WRITE "reason"
==================================================

The "reason" field MUST contain per-word analysis.

You MUST:
- analyze each word in sentence order
- explicitly state the contextual part of speech of each word
- explicitly state whether it is counted into output
- explain why

Use this style inside "reason":
1. word -> 在句中词性 / 是否计入 / 原因
2. word -> 在句中词性 / 是否计入 / 原因
3. word -> 在句中词性 / 是否计入 / 原因

Example format:
"1. A->冠词，不计入；2. man->名词，不按动词计入；3. is->助动词，不计入；4. holding->动词，现在分词表示动作，计入；5. bananas->名词，不计入"

The "reason" must NOT be short or vague.
The "reason" must show the actual decision process for each word.

==================================================
TASK-SPECIFIC DEFINITION OF VERB
==================================================

In this dataset, a verb is:
an action word, event word, process word, or result-event word
that describes what someone/something does
or what has happened to it.

Main verb types:
- lexical action verbs: walk, hold, ride, sit, play, eat, catch, get, help, make, take
- eventive -ing forms: walking, holding, riding, sitting, standing, grazing
- result/event participles: painted, decorated, displayed, canned, parked, filled, shown, chopped

==================================================
WHAT IS NOT A VERB IN THIS TASK
==================================================

Do NOT count:
1. auxiliaries:
is, are, was, were, am, be, been, being

2. pure prepositions / particles:
on, in, at, with, near, from, to, into, onto, over, under, of, by, for, through, around,
up, down, off, out, away, back

3. noun-like words:
jump, trick, game, rail, fun, base, pitch, shot

4. ordinary adjectives / states:
full, open, ready, bright, barefoot, sound

==================================================
CONTEXT RULE
==================================================

A word may have multiple parts of speech.
You MUST judge it from the sentence context, not from the word alone.

Examples:
- "jump" in "doing a jump" is a noun, not a counted verb
- "doing" in "doing a jump" is a verb
- "holding" in "a man is holding bananas" is a verb
- "open" in "the door is open" is not a counted verb

==================================================
MAIN VERB RULES
==================================================

1. Never count auxiliaries:
is, are, was, were, be, been, being

2. Never count prepositions or particles:
into, onto, with, on, in, at, under, over, through, around, up, down, off, out

3. Count action/event/result verbs only.

4. In "doing a jump" / "doing a trick" / "having fun":
count the verb, not the noun object.

5. Some main lexical predicates still count even if they are not dynamic actions:
- has / have when meaning contains/features
- contains
- shows
- reads
- holds
- brings
- makes
- seems
- appears

==================================================
FEW-SHOT EXAMPLES
==================================================

Sentence: 'A man is using a cell phone to photograph a barn'
Output:
{"reason":"1. A->冠词，不计入；2. man->名词，不按动词计入；3. is->助动词，不计入；4. using->动词，现在分词表示动作，计入；5. a->冠词，不计入；6. cell->名词，不计入；7. phone->名词，不计入；8. to->不定式标记，不单独计入；9. photograph->动词原形，表示动作，计入；10. a->冠词，不计入；11. barn->名词，不计入","output":["using","photograph"]}

Sentence: 'A skateboarder hitting a trick on a ramp'
Output:
{"reason":"1. A->冠词，不计入；2. skateboarder->名词，不按动词计入；3. hitting->动词，现在分词表示动作，计入；4. a->冠词，不计入；5. trick->名词，在 hitting a trick 中是宾语，不按动词计入；6. on->介词，不计入；7. a->冠词，不计入；8. ramp->名词，不计入","output":["hitting"]}

Sentence: 'A person on a court with a tennis racket'
Output:
{"reason":"1. A->冠词，不计入；2. person->名词，不按动词计入；3. on->介词，不计入；4. a->冠词，不计入；5. court->名词，不计入；6. with->介词，不计入；7. a->冠词，不计入；8. tennis->名词/修饰成分，不按动词计入；9. racket->名词，不计入；整句没有可计数动词","output":[]}

Sentence: 'Jars of food are being canned in boiling water'
Output:
{"reason":"1. Jars->名词，不按动词计入；2. of->介词，不计入；3. food->名词，不计入；4. are->助动词，不计入；5. being->助动词成分，不计入；6. canned->过去分词，表示被处理的事件结果，按数据集计入；7. in->介词，不计入；8. boiling->此处修饰 water，不作为主要计数动词；9. water->名词，不计入","output":["canned"]}

Sentence: 'A fruit and vegetable stand has bananas up front'
Output:
{"reason":"1. A->冠词，不计入；2. fruit->名词，不按动词计入；3. and->连词，不计入；4. vegetable->名词，不按动词计入；5. stand->名词，不按动词计入；6. has->动词，作主句谓语，表示具有/包含，按数据集计入；7. bananas->名词，不计入；8. up->副词/方位成分，不计入；9. front->名词性方位成分，此处不按动词计入","output":["has"]}

Sentence: 'A fish eyed shot of a skateboarder having fun in a park'
Output:
{"reason":"1. A->冠词，不计入；2. fish->修饰成分，不计入；3. eyed->修饰成分，不计入；4. shot->名词，不按动词计入；5. of->介词，不计入；6. a->冠词，不计入；7. skateboarder->名词，不计入；8. having->动词，现在分词表示动作/状态过程，按数据集计入；9. fun->名词，在 having fun 中作宾语，不按动词计入；10. in->介词，不计入；11. a->冠词，不计入；12. park->名词，不计入","output":["having"]}

{"reason":"1. Jars->名词，不计入；2. of->介词，不计入；3. food->名词，不计入；4. are->助动词，不计入；5. being->助动词成分，不计入；6. canned->过去分词，实义动词，计入；7. in->介词，不计入；8. a->冠词，不计入；9. pot->名词，不计入；10. of->介词，不计入；11. boiling->现在分词，作定语修饰water，计入；12. water->名词，不计入","output":["canned","boiling"]}

Sentence: 'A baseball player catches the ball as an opponent makes it on base'
Output:
{"reason":"1. A->冠词，不计入；2. baseball->名词/修饰成分，不计入；3. player->名词，不计入；4. catches->动词，第三人称单数，计入；5. the->冠词，不计入；6. ball->名词，不计入；7. as->连词，不计入；8. an->冠词，不计入；9. opponent->名词，不计入；10. makes->动词，第三人称单数，计入；11. it->代词，不计入；12. on->介词，不计入；13. base->名词，不计入","output":["catches","makes"]}

Sentence: 'Two pieces of pizza with lasagna toppings on a plate'
Output:
{"reason":"1. Two->数词，不计入；2. pieces->名词，不计入；3. of->介词，不计入；4. pizza->名词，不计入；5. with->介词，不计入；6. lasagna->名词/修饰成分，不计入；7. toppings->名词，不计入；8. on->介词，不计入；9. a->冠词，不计入；10. plate->名词，不计入；整句没有可计数动词","output":[]}

Sentence: 'A elephant that is standing on a floor at a bowling alley'
Output:
{"reason":"1. A->冠词，不计入；2. elephant->名词，不计入；3. that->关系代词，不计入；4. is->助动词，不计入；5. standing->现在分词，实义动词，计入；6. on->介词，不计入；7. a->冠词，不计入；8. floor->名词，不计入；9. at->介词，不计入；10. a->冠词，不计入；11. bowling->动名词/形容词化，修饰alley，通常不计入主要动词，但若参照前例可能需确认。在 'bowling alley' 中 bowling 已名词化/形容词化，类似 tennis racket，故不计入；12. alley->名词，不计入","output":["standing"]}

Sentence: 'A kitchen counter with dirty dishes and empty wine bottles on it'
Output:
{"reason":"1. A->冠词，不计入；2. kitchen->名词/修饰成分，不计入；3. counter->名词，不计入；4. with->介词，不计入；5. dirty->形容词，不计入；6. dishes->名词，不计入；7. and->连词，不计入；8. empty->形容词，不计入；9. wine->名词/修饰成分，不计入；10. bottles->名词，不计入；11. on->介词，不计入；12. it->代词，不计入；整句没有可计数动词","output":[]}

Sentence: 'A BOY ON THE LAWN AT A CAMP GROUND FLYING A KITE'
Output:
{"reason":"1. A->冠词，不计入；2. BOY->名词，不计入；3. ON->介词，不计入；4. THE->冠词，不计入；5. LAWN->名词，不计入；6. AT->介词，不计入；7. A->冠词，不计入；8. CAMP->名词/修饰成分，不计入；9. GROUND->名词，不计入；10. FLYING->此处为标题式短语中的分词，但在某些严格语法计数中若无谓语动词则不计，或视为非限定动词。参考数据output为0，说明此处Flying未被计入（可能因缺乏明确的主谓结构或被视为图像描述标签而非完整句子谓语）；11. A->冠词，不计入；12. KITE->名词，不计入","output":[]}

Sentence: 'The plate of soup has sides of meat and vegetables'
Output:
{"reason":"1. The->冠词，不计入；2. plate->名词，不计入；3. of->介词，不计入；4. soup->名词，不计入；5. has->动词，第三人称单数，计入；6. sides->名词，不计入；7. of->介词，不计入；8. meat->名词，不计入；9. and->连词，不计入；10. vegetables->名词，不计入","output":["has"]}

Sentence: 'a person in a red robe is riding a brown and black horse'
Output:
{"reason":"1. a->冠词，不计入；2. person->名词，不计入；3. in->介词，不计入；4. a->冠词，不计入；5. red->形容词，不计入；6. robe->名词，不计入；7. is->助动词，不计入；8. riding->现在分词，实义动词，计入；9. a->冠词，不计入；10. brown->形容词，不计入；11. and->连词，不计入；12. black->形容词，不计入；13. horse->名词，不计入","output":["riding"]}

Sentence: 'A stack of four pancakes on a skillet'
Output:
{"reason":"1. A->冠词，不计入；2. stack->名词，不计入；3. of->介词，不计入；4. four->数词，不计入；5. pancakes->名词，不计入；6. on->介词，不计入；7. a->冠词，不计入；8. skillet->名词，不计入；整句没有可计数动词","output":[]}

Sentence: 'A man showing a boy with a helmet on how to get on a skateboard'
Output:
{"reason":"1. A->冠词，不计入；2. man->名词，不计入；3. showing->现在分词，实义动词，计入；4. a->冠词，不计入；5. boy->名词，不计入；6. with->介词，不计入；7. a->冠词，不计入；8. helmet->名词，不计入；9. on->介词/副词，不计入；10. how->疑问副词，不计入；11. to->不定式标记，不计入；12. get->动词原形，不定式中的实义动词，计入；13. on->介词，不计入；14. a->冠词，不计入；15. skateboard->名词，不计入","output":["showing","get"]}

Sentence: 'several double decker buses on a crowded urban street'
Output:
{"reason":"1. several->形容词/限定词，不计入；2. double->形容词，不计入；3. decker->名词/形容词，不计入；4. buses->名词，不计入；5. on->介词，不计入；6. a->冠词，不计入；7. crowded->形容词，不计入；8. urban->形容词，不计入；9. street->名词，不计入；整句没有可计数动词","output":[]}

Sentence: 'An artistic version of a roller coaster at theme park'
Output:
{"reason":"1. An->冠词，不计入；2. artistic->形容词，不计入；3. version->名词，不计入；4. of->介词，不计入；5. a->冠词，不计入；6. roller->名词/修饰成分，不计入；7. coaster->名词，不计入；8. at->介词，不计入；9. theme->名词/修饰成分，不计入；10. park->名词，不计入；整句没有可计数动词","output":[]}

Sentence: 'A man holds a red frisbee in preparation of throwing it'
Output:
{"reason":"1. A->冠词，不计入；2. man->名词，不计入；3. holds->动词，第三人称单数，计入；4. a->冠词，不计入；5. red->形容词，不计入；6. frisbee->名词，不计入；7. in->介词，不计入；8. preparation->名词，不计入；9. of->介词，不计入；10. throwing->动名词/现在分词，实义动词，计入；11. it->代词，不计入","output":["holds","throwing"]}

{"reason":"1. an->冠词，不计入；2. image->名词，不计入；3. of->介词，不计入；4. a->冠词，不计入；5. man->名词，不计入；6. that->关系代词，不计入；7. is->动词（系动词），计入；8. on->介词，不计入；9. floor->名词，不计入；10. playing->现在分词，实义动词，计入；11. with->介词，不计入；12. child->名词，不计入","output":["is","playing"]}

Sentence: 'An empty looking bathroom is painted two tone beige'
Output:
{"reason":"1. An->冠词，不计入；2. empty->形容词，不计入；3. looking->现在分词，实义动词（作定语或谓语一部分），计入；4. bathroom->名词，不计入；5. is->助动词，不计入；6. painted->过去分词，实义动词，计入；7. two->数词，不计入；8. tone->名词，不计入；9. beige->名词/形容词，不计入","output":["looking","painted"]}

Sentence: 'A man is using a cell phone to photograph a barn'
Output:
{"reason":"1. A->冠词，不计入；2. man->名词，不计入；3. is->助动词，不计入；4. using->现在分词，实义动词，计入；5. a->冠词，不计入；6. cell->名词，不计入；7. phone->名词，不计入；8. to->不定式标记，不计入；9. photograph->动词原形，实义动词，计入；10. a->冠词，不计入；11. barn->名词，不计入","output":["using","photograph"]}

Sentence: 'a sink sits in front of a window and a counter'
Output:
{"reason":"1. a->冠词，不计入；2. sink->名词，不计入；3. sits->动词，第三人称单数，计入；4. in->介词，不计入；5. front->名词，不计入；6. of->介词，不计入；7. a->冠词，不计入；8. window->名词，不计入；9. and->连词，不计入；10. a->冠词，不计入；11. counter->名词，不计入","output":["sits"]}

Sentence: 'A close up view of some tasty looking food'
Output:
{"reason":"1. A->冠词，不计入；2. close->形容词/副词，不计入；3. up->副词，不计入；4. view->名词，不计入；5. of->介词，不计入；6. some->限定词，不计入；7. tasty->形容词，不计入；8. looking->现在分词，实义动词（作定语修饰food），计入；9. food->名词，不计入","output":["looking"]}

Sentence: 'A boy is standing on a chair using the kitchen sink'
Output:
{"reason":"1. A->冠词，不计入；2. boy->名词，不计入；3. is->助动词，不计入；4. standing->现在分词，实义动词，计入；5. on->介词，不计入；6. a->冠词，不计入；7. chair->名词，不计入；8. using->现在分词，实义动词，计入；9. the->冠词，不计入；10. kitchen->名词/修饰成分，不计入；11. sink->名词，不计入","output":["standing","using"]}

Sentence: 'A woman walks with an umbrella over her head'
Output:
{"reason":"1. A->冠词，不计入；2. woman->名词，不计入；3. walks->动词，第三人称单数，计入；4. with->介词，不计入；5. an->冠词，不计入；6. umbrella->名词，不计入；7. over->介词，不计入；8. her->代词，不计入；9. head->名词，不计入","output":["walks"]}

Sentence: 'Three slices of pizza in a box on a table'
Output:
{"reason":"1. Three->数词，不计入；2. slices->名词，不计入；3. of->介词，不计入；4. pizza->名词，不计入；5. in->介词，不计入；6. a->冠词，不计入；7. box->名词，不计入；8. on->介词，不计入；9. a->冠词，不计入；10. table->名词，不计入；整句没有可计数动词","output":[]}

Sentence: 'a tennis player that has missed the ball'
Output:
{"reason":"1. a->冠词，不计入；2. tennis->名词/修饰成分，不计入；3. player->名词，不计入；4. that->关系代词，不计入；5. has->助动词，不计入；6. missed->过去分词，实义动词，计入；7. the->冠词，不计入；8. ball->名词，不计入","output":["missed"]}

Sentence: 'A street sign at the crosswalk of a road'
Output:
{"reason":"1. A->冠词，不计入；2. street->名词/修饰成分，不计入；3. sign->名词，不计入；4. at->介词，不计入；5. the->冠词，不计入；6. crosswalk->名词，不计入；7. of->介词，不计入；8. a->冠词，不计入；9. road->名词，不计入；整句没有可计数动词","output":[]}

Sentence: 'A little boy that is bending over near a suitcase'
Output:
{"reason":"1. A->冠词，不计入；2. little->形容词，不计入；3. boy->名词，不计入；4. that->关系代词，不计入；5. is->助动词，不计入；6. bending->现在分词，实义动词，计入；7. over->副词/介词，不计入；8. near->介词，不计入；9. a->冠词，不计入；10. suitcase->名词，不计入","output":["bending"]}

Sentence: 'a cat that is sitting down on a old car'
Output:
{"reason":"1. a->冠词，不计入；2. cat->名词，不计入；3. that->关系代词，不计入；4. is->助动词，不计入；5. sitting->现在分词，实义动词，计入；6. down->副词，不计入；7. on->介词，不计入；8. a->冠词，不计入；9. old->形容词，不计入；10. car->名词，不计入","output":["sitting"]}

Sentence: 'Many teddy bears are displayed in front of the trees'
Output:
{"reason":"1. Many->限定词，不计入；2. teddy->名词/修饰成分，不计入；3. bears->名词，不计入；4. are->助动词，不计入；5. displayed->过去分词，实义动词（被动语态），计入；6. in->介词，不计入；7. front->名词，不计入；8. of->介词，不计入；9. the->冠词，不计入；10. trees->名词，不计入","output":["displayed"]}

Sentence: 'A coffe and plate of bread sit next to a pillar'
Output:
{"reason":"1. A->冠词，不计入；2. coffe->名词，不计入；3. and->连词，不计入；4. plate->名词，不计入；5. of->介词，不计入；6. bread->名词，不计入；7. sit->动词，第三人称复数，计入；8. next->副词，不计入；9. to->介词，不计入；10. a->冠词，不计入；11. pillar->名词，不计入","output":["sit"]}

Sentence: 'A lone bird perched on a branch in a wooded area'
Output:
{"reason":"1. A->冠词，不计入；2. lone->形容词，不计入；3. bird->名词，不计入；4. perched->过去分词/过去式，实义动词，计入；5. on->介词，不计入；6. a->冠词，不计入；7. branch->名词，不计入；8. in->介词，不计入；9. a->冠词，不计入；10. wooded->形容词，不计入；11. area->名词，不计入","output":["perched"]}

Sentence: 'a person doing a jump with a skateboard next to a ramp'
Output:
{"reason":"1. a->冠词，不计入；2. person->名词，不计入；3. doing->现在分词，实义动词，计入；4. a->冠词，不计入；5. jump->名词，在 doing a jump 中作宾语，不按动词计入；6. with->介词，不计入；7. a->冠词，不计入；8. skateboard->名词，不计入；9. next->副词，不计入；10. to->介词，不计入；11. a->冠词，不计入；12. ramp->名词，不计入","output":["doing"]}

Sentence: 'A trailer cart filled up high with travel luggage'
Output:
{"reason":"1. A->冠词，不计入；2. trailer->名词/修饰成分，不计入；3. cart->名词，不计入；4. filled->过去分词，实义动词，计入；5. up->副词，不计入；6. high->形容词/副词，不计入；7. with->介词，不计入；8. travel->名词/修饰成分，不计入；9. luggage->名词，不计入","output":["filled"]}

Sentence: 'A skateboarder hitting a trick on a ramp'
Output:
{"reason":"1. A->冠词，不计入；2. skateboarder->名词，不计入；3. hitting->现在分词，实义动词，计入；4. a->冠词，不计入；5. trick->名词，不计入；6. on->介词，不计入；7. a->冠词，不计入；8. ramp->名词，不计入","output":["hitting"]}

Sentence: 'two people riding on a motorcycle with buildings in the background'
Output:
{"reason":"1. two->数词，不计入；2. people->名词，不计入；3. riding->现在分词，实义动词，计入；4. on->介词，不计入；5. a->冠词，不计入；6. motorcycle->名词，不计入；7. with->介词，不计入；8. buildings->名词，不计入；9. in->介词，不计入；10. the->冠词，不计入；11. background->名词，不计入","output":["riding"]}

Sentence: 'One person tossing a frisbee to another person in front of some trees'
Output:
{"reason":"1. One->数词/限定词，不计入；2. person->名词，不计入；3. tossing->现在分词，实义动词，计入；4. a->冠词，不计入；5. frisbee->名词，不计入；6. to->介词，不计入；7. another->限定词，不计入；8. person->名词，不计入；9. in->介词，不计入；10. front->名词，不计入；11. of->介词，不计入；12. some->限定词，不计入；13. trees->名词，不计入","output":["tossing"]}

Sentence: 'A herd of elephants walking over a rocky area with trees in the background'
Output:
{"reason":"1. A->冠词，不计入；2. herd->名词，不计入；3. of->介词，不计入；4. elephants->名词，不计入；5. walking->现在分词，实义动词，计入；6. over->介词，不计入；7. a->冠词，不计入；8. rocky->形容词，不计入；9. area->名词，不计入；10. with->介词，不计入；11. trees->名词，不计入；12. in->介词，不计入；13. the->冠词，不计入；14. background->名词，不计入","output":["walking"]}

Sentence: 'An orange and white cat sleeping with its head on the keyboard of a laptop computer'
Output:
{"reason":"1. An->冠词，不计入；2. orange->形容词，不计入；3. and->连词，不计入；4. white->形容词，不计入；5. cat->名词，不计入；6. sleeping->现在分词，实义动词，计入；7. with->介词，不计入；8. its->代词，不计入；9. head->名词，不计入；10. on->介词，不计入；11. the->冠词，不计入；12. keyboard->名词，不计入；13. of->介词，不计入；14. a->冠词，不计入；15. laptop->名词/修饰成分，不计入；16. computer->名词，不计入","output":["sleeping"]}

Sentence: 'A small girl jumps on her twin size bed'
Output:
{"reason":"1. A->冠词，不计入；2. small->形容词，不计入；3. girl->名词，不计入；4. jumps->动词，第三人称单数，计入；5. on->介词，不计入；6. her->代词，不计入；7. twin->名词/修饰成分，不计入；8. size->名词/修饰成分，不计入；9. bed->名词，不计入","output":["jumps"]}

Sentence: 'A small kitchen knife on top of a sliced carrot'
Output:
{"reason":"1. A->冠词，不计入；2. small->形容词，不计入；3. kitchen->名词/修饰成分，不计入；4. knife->名词，不计入；5. on->介词，不计入；6. top->名词，不计入；7. of->介词，不计入；8. a->冠词，不计入；9. sliced->过去分词作形容词修饰carrot，此处视为非谓语形容词性质，不计入主要动词；10. carrot->名词，不计入","output":[]}

Sentence: 'A train on a track in the middle of a neighborhood'
Output:
{"reason":"1. A->冠词，不计入；2. train->名词，不计入；3. on->介词，不计入；4. a->冠词，不计入；5. track->名词，不计入；6. in->介词，不计入；7. the->冠词，不计入；8. middle->名词，不计入；9. of->介词，不计入；10. a->冠词，不计入；11. neighborhood->名词，不计入","output":[]}

Sentence: 'A fish eyed shot of a skateboarder having fun in a park'
Output:
{"reason":"1. A->冠词，不计入；2. fish->名词/修饰成分，不计入；3. eyed->形容词后缀部分，不计入；4. shot->名词，不计入；5. of->介词，不计入；6. a->冠词，不计入；7. skateboarder->名词，不计入；8. having->现在分词，实义动词，计入；9. fun->名词，不计入；10. in->介词，不计入；11. a->冠词，不计入；12. park->名词，不计入","output":["having"]}

Sentence: 'A balck an white photo of skiers outside a house'
Output:
{"reason":"1. A->冠词，不计入；2. balck->形容词，不计入；3. an->冠词，不计入；4. white->形容词，不计入；5. photo->名词，不计入；6. of->介词，不计入；7. skiers->名词，不计入；8. outside->介词，不计入；9. a->冠词，不计入；10. house->名词，不计入","output":[]}

Sentence: 'A group of elephants who are standing in the grass'
Output:
{"reason":"1. A->冠词，不计入；2. group->名词，不计入；3. of->介词，不计入；4. elephants->名词，不计入；5. who->关系代词，不计入；6. are->助动词，不计入；7. standing->现在分词，实义动词，计入；8. in->介词，不计入；9. the->冠词，不计入；10. grass->名词，不计入","output":["standing"]}

Sentence: 'A fruit and vegetable stand has bananas up front'
Output:
{"reason":"1. A->冠词，不计入；2. fruit->名词/修饰成分，不计入；3. and->连词，不计入；4. vegetable->名词/修饰成分，不计入；5. stand->名词，不计入；6. has->动词，第三人称单数，计入；7. bananas->名词，不计入；8. up->副词，不计入；9. front->名词，不计入","output":["has"]}

Sentence: 'A zebra standing in grass in its enclosure'
Output:
{"reason":"1. A->冠词，不计入；2. zebra->名词，不计入；3. standing->现在分词，实义动词，计入；4. in->介词，不计入；5. grass->名词，不计入；6. in->介词，不计入；7. its->代词，不计入；8. enclosure->名词，不计入","output":["standing"]}

Sentence: 'A man dressed in suit in business meeting room'
Output:
{"reason":"1. A->冠词，不计入；2. man->名词，不计入；3. dressed->过去分词，实义动词，计入；4. in->介词，不计入；5. suit->名词，不计入；6. in->介词，不计入；7. business->名词/修饰成分，不计入；8. meeting->动名词/形容词化，修饰room，通常作为定语不计入核心动词，或视为非谓语；在此语境下dressed为主要动作描述；若参照类似结构，meeting常作定语。根据输出为1，故只计dressed；9. room->名词，不计入","output":["dressed"]}

Sentence: 'Dogs and cat sleeping on big comfortable couch'
Output:
{"reason":"1. Dogs->名词，不计入；2. and->连词，不计入；3. cat->名词，不计入；4. sleeping->现在分词，实义动词，计入；5. on->介词，不计入；6. big->形容词，不计入；7. comfortable->形容词，不计入；8. couch->名词，不计入","output":["sleeping"]}

Sentence: 'There is woman texting on a phone holding a tennis racket'
Output:
{"reason":"1. There->代词/引导词，不计入；2. is->助动词，不计入；3. woman->名词，不计入；4. texting->现在分词，实义动词，计入；5. on->介词，不计入；6. a->冠词，不计入；7. phone->名词，不计入；8. holding->现在分词，实义动词，计入；9. a->冠词，不计入；10. tennis->名词/修饰成分，不计入；11. racket->名词，不计入","output":["texting","holding"]}

Sentence: 'A clock tower on the front of a building with a sun dial'
Output:
{"reason":"1. A->冠词，不计入；2. clock->名词/修饰成分，不计入；3. tower->名词，不计入；4. on->介词，不计入；5. the->冠词，不计入；6. front->名词，不计入；7. of->介词，不计入；8. a->冠词，不计入；9. building->名词，不计入；10. with->介词，不计入；11. a->冠词，不计入；12. sun->名词/修饰成分，不计入；13. dial->名词，不计入","output":[]}

Sentence: 'A man with a baby holding a carrot'
Output:
{"reason":"1. A->冠词，不计入；2. man->名词，不计入；3. with->介词，不计入；4. a->冠词，不计入；5. baby->名词，不计入；6. holding->现在分词，实义动词，计入；7. a->冠词，不计入；8. carrot->名词，不计入","output":["holding"]}

Sentence: 'a man is giving a bottle to a dog'
Output:
{"reason":"1. a->冠词，不计入；2. man->名词，不计入；3. is->助动词，不计入；4. giving->现在分词，实义动词，计入；5. a->冠词，不计入；6. bottle->名词，不计入；7. to->介词，不计入；8. a->冠词，不计入；9. dog->名词，不计入","output":["giving"]}

Sentence: 'A man on a horse is doing a jump'
Output:
{"reason":"1. A->冠词，不计入；2. man->名词，不计入；3. on->介词，不计入；4. a->冠词，不计入；5. horse->名词，不计入；6. is->助动词，不计入；7. doing->现在分词，实义动词，计入；8. a->冠词，不计入；9. jump->名词，在 doing a jump 中作宾语，不按动词计入","output":["doing"]}

Sentence: 'A nicely displayed bathroom sink in a hotel'
Output:
{"reason":"1. A->冠词，不计入；2. nicely->副词，不计入；3. displayed->过去分词，实义动词（作定语），计入；4. bathroom->名词/修饰成分，不计入；5. sink->名词，不计入；6. in->介词，不计入；7. a->冠词，不计入；8. hotel->名词，不计入","output":["displayed"]}

Sentence: 'A frumpled bed sits in between two blue covered nightstands'
Output:
{"reason":"1. A->冠词，不计入；2. frumpled->形容词，不计入；3. bed->名词，不计入；4. sits->动词，第三人称单数，计入；5. in->介词，不计入；6. between->介词，不计入；7. two->数词，不计入；8. blue->形容词，不计入；9. covered->过去分词，实义动词（作定语），计入；10. nightstands->名词，不计入","output":["sits","covered"]}

Sentence: 'a colage of a window being closed with a clock above a window'
Output:
{"reason":"1. a->冠词，不计入；2. colage->名词，不计入；3. of->介词，不计入；4. a->冠词，不计入；5. window->名词，不计入；6. being->助动词成分，不计入；7. closed->过去分词，实义动词，计入；8. with->介词，不计入；9. a->冠词，不计入；10. clock->名词，不计入；11. above->介词，不计入；12. a->冠词，不计入；13. window->名词，不计入","output":["closed"]}

Sentence: 'While the pitcher is winding up for the pitch, the runner is ready to react'
Output:
{"reason":"1. While->连词，不计入；2. the->冠词，不计入；3. pitcher->名词，不计入；4. is->助动词，不计入；5. winding->现在分词，实义动词，计入；6. up->副词，不计入；7. for->介词，不计入；8. the->冠词，不计入；9. pitch->名词，不计入；10. the->冠词，不计入；11. runner->名词，不计入；12. is->系动词，此处构成系表结构 'is ready'，通常系动词单独计数时存在争议，但参考数据为2。若只算winding和react则为2？不，react是不定式。若算winding和ready? No. 让我们看结构：主句1 'pitcher is winding', 主句2 'runner is ready to react'. 'winding' 是动词。'react' 是动词原形。'is' 是助动词/系动词。如果 output 是 2，可能是 'winding' 和 'react'。或者 'winding' 和 'ready' (作为形容词化的动词)? 通常 'to react' 中的 react 是实义动词。让我们假设计入 'winding' 和 'react'。注意 'is ready' 中的 is 是系动词，往往不计入或视情况而定。但在 'is winding' 中 is 是助动词。所以核心动作是 winding 和 react。","output":["winding","react"]}

Sentence: 'Two shots of a woman swinging at a tennis ball'
Output:
{"reason":"1. Two->数词，不计入；2. shots->名词，不计入；3. of->介词，不计入；4. a->冠词，不计入；5. woman->名词，不计入；6. swinging->现在分词，实义动词，计入；7. at->介词，不计入；8. a->冠词，不计入；9. tennis->名词/修饰成分，不计入；10. ball->名词，不计入","output":["swinging"]}

Sentence: 'You can still get tacos and burritos late at night from this truck'
Output:
{"reason":"1. You->代词，不计入；2. can->情态动词，不计入；3. still->副词，不计入；4. get->动词原形，实义动词，计入；5. tacos->名词，不计入；6. and->连词，不计入；7. burritos->名词，不计入；8. late->副词/形容词，不计入；9. at->介词，不计入；10. night->名词，不计入；11. from->介词，不计入；12. this->限定词，不计入；13. truck->名词，不计入","output":["get"]}

Sentence: 'A work station with a computer on it and a guitar on the wall'
Output:
{"reason":"1. A->冠词，不计入；2. work->名词/修饰成分，不计入；3. station->名词，不计入；4. with->介词，不计入；5. a->冠词，不计入；6. computer->名词，不计入；7. on->介词，不计入；8. it->代词，不计入；9. and->连词，不计入；10. a->冠词，不计入；11. guitar->名词，不计入；12. on->介词，不计入；13. the->冠词，不计入；14. wall->名词，不计入","output":[]}

Sentence: 'A white toilet sitting in a stall next to a hand rail'
Output:
{"reason":"1. A->冠词，不计入；2. white->形容词，不计入；3. toilet->名词，不计入；4. sitting->现在分词，实义动词，计入；5. in->介词，不计入；6. a->冠词，不计入；7. stall->名词，不计入；8. next->副词，不计入；9. to->介词，不计入；10. a->冠词，不计入；11. hand->名词/修饰成分，不计入；12. rail->名词，不计入","output":["sitting"]}

Sentence: 'a man walking down the street holding a skateboard'
Output:
{"reason":"1. a->冠词，不计入；2. man->名词，不计入；3. walking->现在分词，实义动词，计入；4. down->介词/副词，不计入；5. the->冠词，不计入；6. street->名词，不计入；7. holding->现在分词，实义动词，计入；8. a->冠词，不计入；9. skateboard->名词，不计入","output":["walking","holding"]}

Sentence: 'four portable toilets in a trailer near a city street'
Output:
{"reason":"1. four->数词，不计入；2. portable->形容词，不计入；3. toilets->名词，不计入；4. in->介词，不计入；5. a->冠词，不计入；6. trailer->名词，不计入；7. near->介词，不计入；8. a->冠词，不计入；9. city->名词/修饰成分，不计入；10. street->名词，不计入","output":[]}

Sentence: 'A large number of motorcycles that are parked'
Output:
{"reason":"1. A->冠词，不计入；2. large->形容词，不计入；3. number->名词，不计入；4. of->介词，不计入；5. motorcycles->名词，不计入；6. that->关系代词，不计入；7. are->助动词，不计入；8. parked->过去分词，实义动词，计入","output":["parked"]}

Sentence: 'There are two people in the room with a dog'
Output:
{"reason":"1. There->代词/引导词，不计入；2. are->助动词/系动词，在此处表示存在，通常不计入实义动词计数，或者视为0个实义动词。参考数据output为0，说明are不被计入；3. two->数词，不计入；4. people->名词，不计入；5. in->介词，不计入；6. the->冠词，不计入；7. room->名词，不计入；8. with->介词，不计入；9. a->冠词，不计入；10. dog->名词，不计入","output":[]}

Sentence: 'The glasses in front of the blender are full'
Output:
{"reason":"1. The->冠词，不计入；2. glasses->名词，不计入；3. in->介词，不计入；4. front->名词，不计入；5. of->介词，不计入；6. the->冠词，不计入；7. blender->名词，不计入；8. are->系动词，不计入；9. full->形容词，不计入","output":[]}

Sentence: 'A luggage bag, laptop, cell phone, and money'
Output:
{"reason":"1. A->冠词，不计入；2. luggage->名词/修饰成分，不计入；3. bag->名词，不计入；4. laptop->名词，不计入；5. cell->名词/修饰成分，不计入；6. phone->名词，不计入；7. and->连词，不计入；8. money->名词，不计入","output":[]}

Sentence: 'A train segment stopped on train tracks in a field'
Output:
{"reason":"1. A->冠词，不计入；2. train->名词/修饰成分，不计入；3. segment->名词，不计入；4. stopped->过去分词/过去式，实义动词，计入；5. on->介词，不计入；6. train->名词/修饰成分，不计入；7. tracks->名词，不计入；8. in->介词，不计入；9. a->冠词，不计入；10. field->名词，不计入","output":["stopped"]}

Sentence: 'a boat a larger ship a buoy and water'
Output:
{"reason":"1. a->冠词，不计入；2. boat->名词，不计入；3. a->冠词，不计入；4. larger->形容词，不计入；5. ship->名词，不计入；6. a->冠词，不计入；7. buoy->名词，不计入；8. and->连词，不计入；9. water->名词，不计入","output":[]}

Sentence: 'A bunch of planes flying close with trails of smoke'
Output:
{"reason":"1. A->冠词，不计入；2. bunch->名词，不计入；3. of->介词，不计入；4. planes->名词，不计入；5. flying->现在分词，实义动词，计入；6. close->副词/形容词，不计入；7. with->介词，不计入；8. trails->名词，不计入；9. of->介词，不计入；10. smoke->名词，不计入","output":["flying"]}

Sentence: 'A cup with a banana sitting inside of it'
Output:
{"reason":"1. A->冠词，不计入；2. cup->名词，不计入；3. with->介词，不计入；4. a->冠词，不计入；5. banana->名词，不计入；6. sitting->现在分词，实义动词，计入；7. inside->介词，不计入；8. of->介词，不计入；9. it->代词，不计入","output":["sitting"]}

Sentence: 'A living room has a fire place and a television with furniture'
Output:
{"reason":"1. A->冠词，不计入；2. living->动名词/形容词化，修饰room，通常作为定语不计入核心动词；3. room->名词，不计入；4. has->动词，第三人称单数，计入；5. a->冠词，不计入；6. fire->名词/修饰成分，不计入；7. place->名词，不计入；8. and->连词，不计入；9. a->冠词，不计入；10. television->名词，不计入；11. with->介词，不计入；12. furniture->名词，不计入","output":["has"]}

Sentence: 'a sail boat sitting in the lake outside the city'
Output:
{"reason":"1. a->冠词，不计入；2. sail->名词/修饰成分，不计入；3. boat->名词，不计入；4. sitting->现在分词，实义动词，计入；5. in->介词，不计入；6. the->冠词，不计入；7. lake->名词，不计入；8. outside->介词，不计入；9. the->冠词，不计入；10. city->名词，不计入","output":["sitting"]}

Sentence: 'A baby elephant standing on a lush green field'
Output:
{"reason":"1. A->冠词，不计入；2. baby->名词/修饰成分，不计入；3. elephant->名词，不计入；4. standing->现在分词，实义动词，计入；5. on->介词，不计入；6. a->冠词，不计入；7. lush->形容词，不计入；8. green->形容词，不计入；9. field->名词，不计入","output":["standing"]}

Sentence: 'A few sail boats sitting on the sand of a beach'
Output:
{"reason":"1. A->冠词，不计入；2. few->限定词，不计入；3. sail->名词/修饰成分，不计入；4. boats->名词，不计入；5. sitting->现在分词，实义动词，计入；6. on->介词，不计入；7. the->冠词，不计入；8. sand->名词，不计入；9. of->介词，不计入；10. a->冠词，不计入；11. beach->名词，不计入","output":["sitting"]}

Sentence: 'A shot of a basic kitchen with white cabinets'
Output:
{"reason":"1. A->冠词，不计入；2. shot->名词，不计入；3. of->介词，不计入；4. a->冠词，不计入；5. basic->形容词，不计入；6. kitchen->名词，不计入；7. with->介词，不计入；8. white->形容词，不计入；9. cabinets->名词，不计入","output":[]}

Sentence: 'Group of men in safety gear next to a bus with emergency equipment'
Output:
{"reason":"1. Group->名词，不计入；2. of->介词，不计入；3. men->名词，不计入；4. in->介词，不计入；5. safety->名词/修饰成分，不计入；6. gear->名词，不计入；7. next->副词，不计入；8. to->介词，不计入；9. a->冠词，不计入；10. bus->名词，不计入；11. with->介词，不计入；12. emergency->形容词/修饰成分，不计入；13. equipment->名词，不计入","output":[]}

Sentence: 'a black pan with an unbaked pizza on it'
Output:
{"reason":"1. a->冠词，不计入；2. black->形容词，不计入；3. pan->名词，不计入；4. with->介词，不计入；5. an->冠词，不计入；6. unbaked->过去分词作形容词修饰pizza，此处视为非谓语形容词性质，不计入主要动词；7. pizza->名词，不计入；8. on->介词，不计入；9. it->代词，不计入","output":[]}

Sentence: 'A group of people and many motor bikes'
Output:
{"reason":"1. A->冠词，不计入；2. group->名词，不计入；3. of->介词，不计入；4. people->名词，不计入；5. and->连词，不计入；6. many->限定词，不计入；7. motor->名词/修饰成分，不计入；8. bikes->名词，不计入","output":[]}

Sentence: 'There is a pile of fruit and vegetables on a table'
Output:
{"reason":"1. There->代词/引导词，不计入；2. is->助动词/系动词，在此处表示存在，通常不计入实义动词计数（参考类似结构如 'There are two people...' output 为 0）；3. a->冠词，不计入；4. pile->名词，不计入；5. of->介词，不计入；6. fruit->名词，不计入；7. and->连词，不计入；8. vegetables->名词，不计入；9. on->介词，不计入；10. a->冠词，不计入；11. table->名词，不计入","output":[]}

Sentence: 'A woman with a handbag walking down a sidewalk by a traffic light'
Output:
{"reason":"1. A->冠词，不计入；2. woman->名词，不计入；3. with->介词，不计入；4. a->冠词，不计入；5. handbag->名词，不计入；6. walking->现在分词，实义动词，计入；7. down->介词/副词，不计入；8. a->冠词，不计入；9. sidewalk->名词，不计入；10. by->介词，不计入；11. a->冠词，不计入；12. traffic->名词/修饰成分，不计入；13. light->名词，不计入","output":["walking"]}

Sentence: 'an  image of table setting with food on it'
Output:
{"reason":"1. an->冠词，不计入；2. image->名词，不计入；3. of->介词，不计入；4. table->名词/修饰成分，不计入；5. setting->动名词/现在分词，实义动词（表设置动作），计入；6. with->介词，不计入；7. food->名词，不计入；8. on->介词，不计入；9. it->代词，不计入","output":["setting"]}

Sentence: 'A person sitting in a car holding onto a red clock'
Output:
{"reason":"1. A->冠词，不计入；2. person->名词，不计入；3. sitting->现在分词，实义动词，计入；4. in->介词，不计入；5. a->冠词，不计入；6. car->名词，不计入；7. holding->现在分词，实义动词，计入；8. onto->介词，不计入；9. a->冠词，不计入；10. red->形容词，不计入；11. clock->名词，不计入","output":["sitting","holding"]}

Sentence: 'A transport truck sitting on the side of a building'
Output:
{"reason":"1. A->冠词，不计入；2. transport->名词/修饰成分，不计入；3. truck->名词，不计入；4. sitting->现在分词，实义动词，计入；5. on->介词，不计入；6. the->冠词，不计入；7. side->名词，不计入；8. of->介词，不计入；9. a->冠词，不计入；10. building->名词，不计入","output":["sitting"]}

Sentence: 'Several people looking at books and magazines at an outdoor zine library'
Output:
{"reason":"1. Several->限定词，不计入；2. people->名词，不计入；3. looking->现在分词，实义动词，计入；4. at->介词，不计入；5. books->名词，不计入；6. and->连词，不计入；7. magazines->名词，不计入；8. at->介词，不计入；9. an->冠词，不计入；10. outdoor->形容词，不计入；11. zine->名词/修饰成分，不计入；12. library->名词，不计入","output":["looking"]}

Sentence: 'A photo taken from a vehicle looking at an intersection'
Output:
{"reason":"1. A->冠词，不计入；2. photo->名词，不计入；3. taken->过去分词，实义动词，计入；4. from->介词，不计入；5. a->冠词，不计入；6. vehicle->名词，不计入；7. looking->现在分词，实义动词，计入；8. at->介词，不计入；9. an->冠词，不计入；10. intersection->名词，不计入","output":["taken","looking"]}

Sentence: 'A large raw carrot and cut up garlic on a cutting board with a knife'
Output:
{"reason":"1. A->冠词，不计入；2. large->形容词，不计入；3. raw->形容词，不计入；4. carrot->名词，不计入；5. and->连词，不计入；6. cut->过去分词，实义动词（cut up 中的核心动词部分），计入；7. up->副词，不计入；8. garlic->名词，不计入；9. on->介词，不计入；10. a->冠词，不计入；11. cutting->动名词/形容词化，修饰board，通常作为定语不计入核心动词；若计为动词则会有歧义，但根据输出1，故只计cut；12. board->名词，不计入；13. with->介词，不计入；14. a->冠词，不计入；15. knife->名词，不计入","output":["cut"]}

Sentence: 'A man exiting a small blue triple decker bus'
Output:
{"reason":"1. A->冠词，不计入；2. man->名词，不计入；3. exiting->现在分词，实义动词，计入；4. a->冠词，不计入；5. small->形容词，不计入；6. blue->形容词，不计入；7. triple->数词/形容词，不计入；8. decker->名词/修饰成分，不计入；9. bus->名词，不计入","output":["exiting"]}

Sentence: 'A living room with chairs and a couch'
Output:
{"reason":"1. A->冠词，不计入；2. living->动名词/形容词化，修饰room，通常作为定语不计入核心动词；3. room->名词，不计入；4. with->介词，不计入；5. chairs->名词，不计入；6. and->连词，不计入；7. a->冠词，不计入；8. couch->名词，不计入","output":[]}

Sentence: 'a close up of three tooth brushes on a sink'
Output:
{"reason":"1. a->冠词，不计入；2. close->形容词/副词，不计入；3. up->副词，不计入；4. of->介词，不计入；5. three->数词，不计入；6. tooth->名词/修饰成分，不计入；7. brushes->名词，不计入；8. on->介词，不计入；9. a->冠词，不计入；10. sink->名词，不计入","output":[]}

==================================================
FINAL INSTRUCTION
==================================================
The number of analysis items in "reason" should match the number of words/tokens in the sentence as closely as possible.
You MUST analyze every word in order inside "reason".
Do NOT skip words.
Do NOT write a short summary.
Do NOT just say "根据上下文判断".
Show the actual per-word decision process.

Return ONLY:
{"reason":"逐词分析过程","output":["...","..."]}
"""

def normalize_prediction(text: str) -> str:
    text = str(text).strip()
    nums = re.findall(r"\d+", text)
    if nums:
        return nums[0]
    return "0"


def process_single_example(sample, noun_system_prompt, verb_system_prompt, task_id, model_name):
    sample_id = sample.get("id", "")
    input_text = sample["input"]
    gt = str(sample["output"][0]).strip()

    lowered = input_text.lower()
    if "count the number of nouns" in lowered:
        system_prompt = noun_system_prompt
        task_type = "nouns"
    elif "count the number of verbs" in lowered:
        system_prompt = verb_system_prompt
        task_type = "verbs"
    else:
        system_prompt = noun_system_prompt
        task_type = "unknown_default_nouns"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_text}
    ]

    raw_output = qwen_api(messages, model=model_name)
    parsed_items, parsed_reason = parse_output_items(raw_output)
    pred = str(len(parsed_items))
    is_correct = (pred == gt)

    return {
        "task_id": task_id,
        "sample_id": sample_id,
        "input": input_text,
        "task_type": task_type,
        "gt": gt,
        "model_output": pred,
        "parsed_items": parsed_items,
        "parsed_reason": parsed_reason,
        "raw_output": raw_output,
        "is_correct": is_correct
    }

def process_single_test(sample, noun_system_prompt, verb_system_prompt, task_id, model_name):
    sample_id = sample.get("id", "")
    input_text = sample["input"]

    lowered = input_text.lower()
    if "count the number of nouns" in lowered:
        system_prompt = noun_system_prompt
        task_type = "nouns"
    elif "count the number of verbs" in lowered:
        system_prompt = verb_system_prompt
        task_type = "verbs"
    else:
        system_prompt = noun_system_prompt
        task_type = "unknown_default_nouns"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_text}
    ]

    raw_output = qwen_api(messages, model=model_name)
    items, parsed_reason = parse_output_items(raw_output)
    pred = len(items)

    return {
        "task_id": task_id,
        "sample_id": sample_id,
        "input": input_text,
        "task_type": task_type,
        "model_output": pred,
        "parsed_items": items,
        "parsed_reason": parsed_reason,
        "raw_output": raw_output
    }

if __name__ == "__main__":
    file_path = r"D:\WorkSpace\python\flagOS赛题三\LongContext-ICL-Annotation\data\openseek-2_count_nouns_verbs.json"

    # 训练集验证结果
    timeFlag = time.strftime("%H%M%S", time.localtime())

    # 测试集提交文件
    test_jsonl_path = r"D:\WorkSpace\python\flagOS赛题三\LongContext-ICL-Annotation\rgs_q2\experiment\openseek-2-v1.jsonl"

    model_name = "/Qwen3-4B/Qwen/Qwen3-4B"

    task_id, task_name, definition_list, examples, test_samples = task2_data_loader(file_path)

    print(f"任务: {task_id} / {task_name}")
    print(f"训练样例数: {len(examples)}")
    print(f"测试样例数: {len(test_samples)}")

    max_workers = 200

    # =========================
    # 1) 训练集验证
    # =========================
    if False:  # 先注释掉训练集验证，等测试集预测分析完再开
        print("\n开始验证训练集（examples）...")

        eval_func = partial(
            process_single_example,
            noun_system_prompt=NOUN_SYSTEM_PROMPT,
            verb_system_prompt=VERB_SYSTEM_PROMPT,
            task_id=task_id,
            model_name=model_name
        )


        eval_results = []
        correct_cnt = 0
        total_cnt = len(examples)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in tqdm(executor.map(eval_func, examples), total=total_cnt, desc="Evaluating examples"):
                
                if result["is_correct"]:
                    correct_cnt += 1
                else:
                    eval_results.append(result)

        accuracy = correct_cnt / total_cnt if total_cnt > 0 else 0.0
        wrong_cases = [x for x in eval_results if not x["is_correct"]]

        print(f"训练集验证完成，总数: {total_cnt}")
        print(f"训练集准确率: {accuracy:.4f}")
        print(f"错误样本数: {len(wrong_cases)}")

        eval_output_data = {
            "task_id": task_id,
            "task_name": task_name,
            "evaluate_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "model_name": model_name,
            "prompt": NOUN_SYSTEM_PROMPT + VERB_SYSTEM_PROMPT,
            "examples_total": total_cnt,
            "examples_correct": correct_cnt,
            "examples_accuracy": accuracy,
            "wrong_cases": wrong_cases,
            "all_results": eval_results
        }

        accuracy = eval_output_data["examples_accuracy"]
        eval_out_path = f"D:/work_files/python_project/flagOS赛题三/LongContext-ICL-Annotation/rgs_q2/experiment/openseek-2_{timeFlag}_acc_{accuracy}.json"

        os.makedirs(os.path.dirname(eval_out_path), exist_ok=True)
        with open(eval_out_path, 'w', encoding='utf-8') as f:
            json.dump(eval_output_data, f, ensure_ascii=False, indent=4)

        print(f"训练集验证结果已保存: {eval_out_path}")

    # =========================
    # 2) 测试集预测
    # =========================
    if True:  # 先注释掉测试集预测，等验证结果分析完再开
        print("\n开始预测测试集（test_samples）...")

        test_func = partial(
            process_single_test,
            noun_system_prompt=NOUN_SYSTEM_PROMPT,
            verb_system_prompt=VERB_SYSTEM_PROMPT,
            task_id=task_id,
            model_name=model_name
        )
        test_results = []
        test_total = len(test_samples)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in tqdm(executor.map(test_func, test_samples), total=test_total, desc="Predicting test"):
                test_results.append(result)

        output_data = [
            {
                "test_sample_id": item["sample_id"],
                "prediction": str(item["model_output"])
            }
            for item in test_results
        ]

        os.makedirs(os.path.dirname(test_jsonl_path), exist_ok=True)
        with open(test_jsonl_path, 'w', encoding='utf-8') as f:
            for item in output_data:
                line = json.dumps(item, ensure_ascii=False)
                f.write(line + '\n')

        print(f"测试集提交文件已保存为 JSONL: {test_jsonl_path}")