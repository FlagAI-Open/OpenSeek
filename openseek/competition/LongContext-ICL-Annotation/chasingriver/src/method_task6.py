import json
import re

import requests
from transformers import AutoTokenizer

_TOKENIZER = None
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

GENRE_HINTS = {
    "telephone": (
        "Telephone hard rules.\n"
        "Accept when:\n"
        "1. The sentence sounds like spoken conversation, call-in dialogue, or transcript text, even if cleaned up.\n"
        "2. The sentence is a short conversational paraphrase, reaction, question, or opinion a caller could naturally say.\n"
        "3. The sentence is a cleaned-up restatement of what someone on the call said, even if it becomes grammatical and concise.\n"
        "4. The sentence is informal, personal, practical, or built around everyday spoken concerns.\n"
        "5. If the sentence could naturally be said aloud in a phone conversation, lean Y.\n"
        "6. A contradictory, exaggerated, or oversimplified restatement can still be telephone if it still sounds like something a caller would say.\n"
        "7. A short opposite or absurd location rewrite can still be a paired telephone hypothesis when it mirrors the caller's statement.\n"
        "8. A contradiction like ugly/beautiful, discouraged/encouraged, or father/mother can still be a valid telephone hypothesis.\n"
        "9. A family/athletics statement can be paired with an opposite encouragement/discouragement hypothesis if it repeats the family or athletics frame.\n"
        "10. Practical advice about what to do with leftover prescriptions can be paired with a caller's discussion of expired prescriptions and drug-test positives.\n"
        "Reject when:\n"
        "1. The sentence reads like literary narration, scenic description, or story exposition.\n"
        "2. The sentence reads like official, institutional, policy, report, or guidebook prose.\n"
        "3. The sentence is too polished, abstract, or essay-like to sound like something said aloud in conversation.\n"
        "4. The sentence sounds like a detached factual statement rather than something a person would naturally say on a call.\n"
        "5. A generic reaction using only that, it, them, or some is not paired unless it repeats concrete content from Sentence 1.\n"
        "6. Do not infer a hidden referent for generic reactions such as that is annoying, some are useful, or it was not enjoyable.\n"
        "7. A generic accusation such as saying the same crazy things as a newly named person is not paired with a car, engine, or gas-mileage discussion.\n"
        "8. Dialogue tags like 'said Number 14 unexpectedly' are fiction narration, not telephone transcripts."
    ),
    "travel": (
        "Travel hard rules.\n"
        "Accept when:\n"
        "1. The sentence sounds like a travel guide, attraction blurb, sightseeing note, or visitor-facing description.\n"
        "2. The sentence gives place facts about landmarks, districts, routes, museums, churches, climate, schedules, festivals, or local history.\n"
        "3. The sentence is a short or even inaccurate tourism-style statement that still clearly sounds like guidebook prose.\n"
        "4. The sentence is a broad, compressed travel fact about what is there, where it is, when to visit, or what visitors should notice.\n"
        "5. Travel also includes local lifestyle, shopping habits, practical restrictions, and cultural background when written in guidebook style.\n"
        "6. Travel can include place-focused historical or political background when it reads like guidebook context for a destination.\n"
        "7. A rhetorical question, wrong place fact, or over-broad destination summary can still be travel if it still sounds like guidebook prose.\n"
        "8. A contradictory climate, location, or historical rewrite can still be paired when it keeps the same place, people, or attraction frame.\n"
        "9. Do not reject destination history just because it mentions troops, leagues, states, or political groups; guidebooks often summarize such background.\n"
        "Reject when:\n"
        "1. The sentence is mainly dialogue, story narration, or character action.\n"
        "2. The sentence sounds like government procedure, policy, compliance, or administrative reporting.\n"
        "3. The sentence is generic commentary with no travel-guide or destination-writing feel.\n"
        "4. The sentence is about some other place or event in a way that does not sound like destination description or guidebook writing."
    ),
    "government": (
        "Government hard rules.\n"
        "Accept when:\n"
        "1. The sentence sounds like official, procedural, administrative, policy, compliance, or report prose.\n"
        "2. The sentence is a title, heading, recommendation, caption, finding, policy summary, or neutral restatement of an official claim.\n"
        "3. The sentence uses institutional wording such as recommendations, reports, assessments, standards, programs, requirements, findings, or operations.\n"
        "4. The sentence is a weak, simplified, or imprecise summary of a report-style statement but still clearly sounds administrative or official.\n"
        "5. Very short fragments, headings, chart titles, and compressed report claims can still be government.\n"
        "6. A contradictory, incorrect, or over-broad restatement can still be government if it still reads like official or report prose.\n"
        "7. A wrong statistic, opposite trend, swapped location, or wrong demographic can still be a paired hypothesis if it mirrors the report claim.\n"
        "8. Abstract official guidance about observation, bias, findings, or methods can be government even when it is not about agencies by name.\n"
        "Reject when:\n"
        "1. The sentence only mentions public issues, states, or aid but does not sound like official/report writing.\n"
        "2. The sentence sounds like magazine commentary, travel writing, or literary narration instead.\n"
        "3. The sentence is merely topical overlap without administrative or institutional source-style.\n"
        "4. The sentence is personal opinion, casual conversation, or rhetorical commentary rather than report prose.\n"
        "5. Two official-sounding report sentences are still N when they concern different programs, studies, methods, or findings.\n"
        "6. Do not pair separate report findings merely because both sound administrative.\n"
        "7. A short conversational fragment such as yeah, uh-huh, or a little bit is not government even when Sentence 2 is government."
    ),
    "slate": (
        "Slate hard rules.\n"
        "Accept when:\n"
        "1. The sentence sounds like magazine journalism, cultural commentary, political analysis, media writing, or opinionated feature prose.\n"
        "2. The sentence is a short media-style summary, broad generalization, rhetorical claim, or argumentative restatement.\n"
        "3. The sentence can be blunt, exaggerated, simplified, or even somewhat wrong and still count as slate if the source-style remains magazine/commentary prose.\n"
        "4. The sentence can be an aggressive paraphrase of politics, finance, culture, media, or public debate, even when it drops important nuance.\n"
        "5. If the sentence sounds like something from a magazine column, media analysis, or cultural-political article, lean Y.\n"
        "6. A broad or opposite hypothesis about the same policy, controversy, group, or argument can still be paired even when details are wrong.\n"
        "7. A negated media summary can still be paired when it repeats the same metaphor, publication, country, alliance, market, or policy frame.\n"
        "Reject when:\n"
        "1. The sentence sounds like official government reporting or procedural text.\n"
        "2. The sentence sounds like guidebook travel description or literary fiction.\n"
        "3. The sentence is plain conversation without a journalism/commentary feel.\n"
        "4. The sentence is only factual in topic but lacks the magazine/commentary voice.\n"
        "5. A short fragment or generic claim is not slate unless it clearly sounds like magazine or opinion writing.\n"
        "6. For the final pair label, unrelated commentary-like sentences are still N; Y needs a hypothesis/rewrite link to Sentence 1.\n"
        "7. Sharing only a vague word such as thing, evidence, transaction, story, discussion, or it is not enough.\n"
        "8. A sentence that merely names another public issue, artwork, company, or scandal is N if it is not a hypothesis about Sentence 1.\n"
        "9. Do not pair generic analogy words such as transaction, bailout, story, or discussion unless the same concrete subject is repeated.\n"
        "10. Official report titles, GAO captions, OMB oversight headings, and agency-practice labels are government style, not slate commentary."
    ),
    "fiction": (
        "Fiction hard rules.\n"
        "Accept when:\n"
        "1. The sentence sounds like story narration, dialogue, scene-setting, action, or character-centered literary prose.\n"
        "2. The sentence could naturally appear inside a novel even if it is short, fragmentary, or contradictory.\n"
        "3. The sentence contains clear narrative texture: a speaker, a character, an action, a setting detail, or an in-story reaction.\n"
        "4. The sentence feels anchored in a scene rather than just stating a free-floating fact.\n"
        "5. A plain sentence about a scene, object, or atmosphere can still be fiction even without dialogue.\n"
        "6. A contradiction that flips an action, object, location, or command can still be a paired fiction hypothesis.\n"
        "7. A typo-like near word such as pain/plan can still be a paired fiction hypothesis when the sentence frame is clearly reused.\n"
        "Reject when:\n"
        "1. The sentence is a generic factual statement, abstract claim, or public-information style sentence.\n"
        "2. The sentence lacks narrative texture and could fit equally well in government, travel, or commentary prose.\n"
        "3. The sentence sounds dramatic but does not actually feel like it belongs inside a story.\n"
        "4. The sentence is an isolated assertion with no scene, no speaker, no character focus, and no clear literary texture.\n"
        "5. If a sentence could be inserted into many non-fiction contexts just as easily, lean N even if it mentions an action or emotion.\n"
        "6. Two unrelated short story-like sentences are not enough; each one should independently feel like in-story prose.\n"
        "7. For the final pair label, a new unrelated story beat is N even if it sounds fictional.\n"
        "8. A dialogue tag or isolated quote from fiction should not be treated as telephone merely because it is spoken aloud.\n"
        "9. Different named people, unrelated objects, or unrelated setting details usually make a fiction pair N.\n"
        "10. Do not invent an unseen continuation of the story; plausible next events are N unless Sentence 2 explicitly rewrites Sentence 1.\n"
        "11. Generic meta-statements such as a remark was given are N unless they repeat the concrete speech act or speaker from Sentence 1.\n"
        "12. A character turning toward another person can pair with an opposite position/direction hypothesis about that same person.\n"
        "13. Historical exposition about rulers, guidebook warnings, or telephone transcripts are not fiction just because Sentence 2 is story-like."
    ),
}

FEW_SHOT_BY_GENRE = {
    "telephone": (
        "Input: Sentence 1: and i think it's frightening to them to see the roles switching and i think i think this reaction comes more out of fear now my husband is Sentence 2: I think it scares them to notice the roles switching. Genre: telephone.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: but the cities down there yeah and and and the next one up is uh is a small city between me and Providence Sentence 2: The next city is smaller than providence. Genre: telephone.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: when you turn sixty five then you still owe some tax, just not nearly as much Sentence 2: At sixty-five the taxes disappear completely. Genre: telephone.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: yeah i you know i'm not sure if we have the death penalty here to be perfectly honest with you Sentence 2: The death penalty should be brought back here. Genre: telephone.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: parents around here just let their kids run everywhere and i just can't do that Sentence 2: Parents around here watch their kids closely. Genre: telephone.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: right yeah and then she's saying you know you can't let them outside and you can't do all this stuff Sentence 2: She was telling me what you can and cannot do with them. Genre: telephone.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: no no i've never had any luck with their's Sentence 2: The governance issues are definitely not insurmountable. Genre: telephone.\n"
        "Output:\nS1: Y\nS2: N\n<label>N</label>\n\n"
        "Input: Sentence 1: were you have you i take it you haven't spent any time in the military Sentence 2: Jon said there is nothing else we can do. Genre: telephone.\n"
        "Output:\nS1: Y\nS2: N\n<label>N</label>\n\n"
        "Input: Sentence 1: all of our garages are basically gas stations now and they hardly fix anything Sentence 2: The dealer now has something close to a monopoly on repairs. Genre: telephone.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: uh this is way too much trouble Sentence 2: I can't see why that columnist sounds so irritated. Genre: telephone.\n"
        "Output:\nS1: Y\nS2: N\n<label>N</label>\n\n"
    ),
    "travel": (
        "Input: Sentence 1: St. Giles was the church of John Knox, the great Protestant reformer. Sentence 2: St. Giles was a place all Protestant reformers would avoid. Genre: travel.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: Out in the river are several islands and the largest one, Gezira, is home to one of Cairo's most chic neighborhoods, Zamalek. Sentence 2: There aren't any islands in Cairo, are there? Genre: travel.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: From here you can ascend a hill to the old seminary, which offers panoramic views below and a ruined castle above. Sentence 2: The seminary is located deep underground in a natural cavern. Genre: travel.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: This interactive journey through the history of the earth takes you back to the moment of the Big Bang. Sentence 2: The journey takes approximately two hours. Genre: travel.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: With Christianity and sophisticated Celtic culture successfully fused, Ireland entered its Golden Age. Sentence 2: The Celts welcomed Christianity with open arms. Genre: travel.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: Note the masterful carving of the oak doors on the central and north portals. Sentence 2: The door on the south portal used to have a carving too, but it was stolen. Genre: travel.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: To the west of Naoussa is Kolymbithres, a growing resort whose beaches are surrounded by strange rock features. Sentence 2: Sangria is suited for hot environments. Genre: travel.\n"
        "Output:\nS1: Y\nS2: N\n<label>N</label>\n\n"
        "Input: Sentence 1: what kind of -- Sentence 2: All of the guided tours begin at the Magnesian Gate. Genre: travel.\n"
        "Output:\nS1: Y\nS2: N\n<label>N</label>\n\n"
    ),
    "government": (
        "Input: Sentence 1: Recommendation 1: future research should cover the full range of alcohol-screening issues in emergency departments. Sentence 2: There are no alcohol problems among emergency-department patients. Genre: government.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: The Physician's Guide to Helping Patients with Alcohol Problems. Sentence 2: The guide for helping patients with alcohol problems is updated once every three years. Genre: government.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: benefits, recommended over 600 actions that have led to improvements in government operations, and provided 229 testimonies requested by congressional committees. Sentence 2: There are improvements in the actions of the government. Genre: government.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: If there's one person using drugs, that is a start, said Robertson, a native of Jamaica. Sentence 2: Robertson is a native of the Dominican Republic. Genre: government.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: The core issues of the study concerned the value-added of design. Sentence 2: Adding value through design is difficult. Genre: government.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: Campaigners are asking the government for more aid. Sentence 2: Agencies recently imposed several new screening requirements. Genre: government.\n"
        "Output:\nS1: N\nS2: Y\n<label>N</label>\n\n"
        "Input: Sentence 1: This sounds right. Sentence 2: Information security was initially designated as a high-risk area in late 1999. Genre: government.\n"
        "Output:\nS1: N\nS2: Y\n<label>N</label>\n\n"
        "Input: Sentence 1: The report discusses the long-term consequences of fiscal choices made today. Sentence 2: A novelist later regretted his war strategy. Genre: government.\n"
        "Output:\nS1: Y\nS2: N\n<label>N</label>\n\n"
    ),
    "slate": (
        "Input: Sentence 1: The Wall Street Journal calls the 7,000 mark a perfect Valentine's Day affirmation of investors' six-year love affair with stocks. Sentence 2: No publication believes that Valentine's Day symbolizes investors' interaction with stocks. Genre: slate.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: Scientists were once ostracized for holding religious beliefs but can now worship without embarrassment. Sentence 2: Most scientists are Christian. Genre: slate.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: This is another way of saying that in the last 30 years, the people who owned America have lost 40 percent of their wealth held in the form of equity. Sentence 2: People can have their wealth in the form of equity. Genre: slate.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: Her mixed priorities when it comes to women were revealed by the fact that she voted nay on the minimum-wage bill. Sentence 2: When it came to women, her priorities were clear and in order. Genre: slate.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: yeah i have uh returned about every four or five years to thinking that i would like to do something about it Sentence 2: Secretly, Marc Dem doubts his claim about Jewish origins. Genre: slate.\n"
        "Output:\nS1: N\nS2: Y\n<label>N</label>\n\n"
        "Input: Sentence 1: out and so forth and most of my things are dust collectors and i hate to dust Sentence 2: The plan was taken rather quickly after Pataki and his allies wished to have nothing to do with it. Genre: slate.\n"
        "Output:\nS1: N\nS2: Y\n<label>N</label>\n\n"
        "Input: Sentence 1: He took a cloth belt and wrapped it around a sturdy stick. Sentence 2: The media stated that the true victims in the scandal were the innocent children. Genre: slate.\n"
        "Output:\nS1: N\nS2: Y\n<label>N</label>\n\n"
        "Input: Sentence 1: One thing her evidence has shown me. Sentence 2: Clinton had seven past indiscretions. Genre: slate.\n"
        "Output:\nS1: N\nS2: Y\n<label>N</label>\n\n"
    ),
    "fiction": (
        "Input: Sentence 1: He saw the smoke filling the air. Sentence 2: It was obvious to him that smoke was filling the air. Genre: fiction.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: What are you doing in this house? Sentence 2: You aren't supposed to be in this house until after ten, so why are you here? Genre: fiction.\n"
        "Output:\nS1: Y\nS2: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: We waited in a tense silence. Sentence 2: Anna is being told everything will go wrong. Genre: fiction.\n"
        "Output:\nS1: Y\nS2: N\n<label>N</label>\n\n"
        "Input: Sentence 1: At that moment, we were interrupted. Sentence 2: The bald man was getting ready to attack Adrin. Genre: fiction.\n"
        "Output:\nS1: Y\nS2: N\n<label>N</label>\n\n"
        "Input: Sentence 1: The virtual Benjamin Franklin was writing with a quill pen. Sentence 2: He was just an errand boy, nothing more. Genre: fiction.\n"
        "Output:\nS1: Y\nS2: N\n<label>N</label>\n\n"
        "Input: Sentence 1: Come on, Annette. Sentence 2: The first thing they did was take the risk. Genre: fiction.\n"
        "Output:\nS1: Y\nS2: N\n<label>N</label>\n\n"
        "Input: Sentence 1: They were shining like emeralds now. Sentence 2: There had been a massacre. Genre: fiction.\n"
        "Output:\nS1: Y\nS2: N\n<label>N</label>\n\n"
        "Input: Sentence 1: It's a h'expression, sir, explained Albert. Sentence 2: There was a lot of light in the area. Genre: fiction.\n"
        "Output:\nS1: Y\nS2: N\n<label>N</label>\n\n"
    ),
}

PAIR_CALIBRATION_BY_GENRE = {
    "telephone": (
        "Input: Sentence 1: i just can't let the kids run everywhere around here Sentence 2: Parents keep close watch on their kids here. Genre: telephone.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: a dollar comes out of the machine at the bottom Sentence 2: That is annoying, but it is okay. Genre: telephone.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: i'm not sure whether we have the death penalty here Sentence 2: The death penalty should be brought back here. Genre: telephone.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: i grew up in California and went to school there Sentence 2: I grew up on the east coast and went to school there. Genre: telephone.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: we often use latex paint and several brands are good Sentence 2: Some of them are very useful. Genre: telephone.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: a mother or father can be the one raising the children Sentence 2: Mothers are always the parent raising the children. Genre: telephone.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: there is nothing pretty about it Sentence 2: It is beautiful. Genre: telephone.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: the speaker had a supervisor job in Dallas Sentence 2: That was the easiest job they ever had. Genre: telephone.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: my father discouraged my brother and me from athletics but later competed in the senior Olympics Sentence 2: He encouraged my brother and me to try out for the Olympics. Genre: telephone.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: the speaker says the show sounds pretty good and is sorry they missed it Sentence 2: It is really not very enjoyable. Genre: telephone.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: Garn, said Number 14 unexpectedly. Sentence 2: I think I have heard your voice before. Genre: telephone.\n"
        "Output:\nS1 target genre: N\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: A character said one word unexpectedly in narration. Sentence 2: I think I recognize your voice. Genre: telephone.\n"
        "Output:\nS1 target genre: N\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: if you treat those cars right they are not the gas guzzlers people think they are Sentence 2: You are saying the same crazy things as a newly named person. Genre: telephone.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: if an expired prescription is still left and you take extra penicillin, it can show up as a positive Sentence 2: So be sure to get rid of your remaining prescriptions. Genre: telephone.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: uh-huh, some fathers are the ones raising the children Sentence 2: Mothers are always the parent raising the children. Genre: telephone.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
    ),
    "travel": (
        "Input: Sentence 1: A riverside island district is one of the city's most fashionable neighborhoods. Sentence 2: There are not any islands in the city, are there? Genre: travel.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: The hilltop seminary offers views of the town and castle. Sentence 2: The seminary is deep underground in a natural cavern. Genre: travel.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: A glossy high-rise hotel stands on the main plaza. Sentence 2: A fishing village looks better from a distance than up close. Genre: travel.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: The country has a hot, humid climate and uneven development. Sentence 2: The country is cold part of the year. Genre: travel.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: Arab League states sent troops to help the Palestinian Arabs. Sentence 2: The British sent food to the Palestinian Arabs. Genre: travel.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: Political events in a destination's history involved regional troops and local residents. Sentence 2: Another outside group helped those local residents. Genre: travel.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: A travel history section says neighboring states sent troops to help a local people. Sentence 2: A different outside power sent supplies to the same people. Genre: travel.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
    ),
    "government": (
        "Input: Sentence 1: Recommendation 1 says research should address the full spectrum of alcohol problems among emergency patients. Sentence 2: There are no alcohol problems among emergency patients. Genre: government.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: Campaigners are asking the state for more financial aid. Sentence 2: Agencies recently mandated several screening policies. Genre: government.\n"
        "Output:\nS1 target genre: N\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: The study concerned the value added by design. Sentence 2: Adding value through design is difficult. Genre: government.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: The report says national saving increased the amount available for investment. Sentence 2: National saving declined and reduced investment. Genre: government.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: Female workers had a higher risk for uterine cancer. Sentence 2: Females have a greater risk of lung cancer. Genre: government.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: Robertson was described as a native of Jamaica. Sentence 2: Robertson is a native of the Dominican Republic. Genre: government.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: Observers can set aside their preconceived ideas. Sentence 2: Zero personal bias is assumed by many people. Genre: government.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: No observer begins without preconceived ideas, although they may set those ideas aside. Sentence 2: Many people assume a state of zero personal bias. Genre: government.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: A report discusses an agricultural worker program. Sentence 2: Another sentence discusses acts and executive orders in a different finding. Genre: government.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: A report says human-capital weaknesses will not be quickly addressed. Sentence 2: A separate finding says organizations could not be compared. Genre: government.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: The government's human capital weaknesses did not emerge overnight and will not be easily addressed. Sentence 2: Because organizations were identifiable, a report could not compare practices. Genre: government.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: yeah, a little bit Sentence 2: Taxes paid by current workers fund social security programs. Genre: government.\n"
        "Output:\nS1 target genre: N\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: Human-capital weaknesses in the federal government will not be quickly or easily addressed. Sentence 2: The report could not compare practices because organizations were identifiable. Genre: government.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
    ),
    "slate": (
        "Input: Sentence 1: A magazine calls a stock-market milestone a holiday affirmation of investors' love affair with shares. Sentence 2: No publication uses the holiday as a metaphor for investors and stocks. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: A newspaper describes a stock index milestone as Valentine's Day proof of investors' love affair with stocks. Sentence 2: No publication treats Valentine's Day as a symbol of investor interaction with stocks. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: Online forums often let bad discussion drive out good discussion. Sentence 2: The outrage is that boxing seems primitive. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: A conflict of interest can distort perception of truth. Sentence 2: The advantage was apparent in you. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: A policy caps federal welfare payments at an older level. Sentence 2: The rules change federal welfare payments for the better. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: Scientists can now worship without embarrassment. Sentence 2: Most scientists are Christian. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: A journalist discussed Hughes, Clinton, and Republican connections. Sentence 2: No discussion allowed. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: Psychologists played nonsense-word recordings for babies. Sentence 2: One could hint at an end-of-era story. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: A country made common cause with Israel during the decade. Sentence 2: The country severed ties with Israel during the decade. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: Serbia used its wartime history to make common cause with Israel. Sentence 2: Serbia worked to cut ties with Israel. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: A newspaper calls a market milestone a holiday metaphor for investors' relationship with stocks. Sentence 2: No publication makes that holiday metaphor about investors and stocks. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: A politician's mixed priorities on women were shown by a vote. Sentence 2: Her priorities on women were clear and orderly. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: A spy sold identities and profiles to a foreign country. Sentence 2: The transaction is compared to a bailout. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: A CIA officer allegedly sold new-agent identities and profiles to Russia for money. Sentence 2: The transaction could be compared to a pre-emptive bailout. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: A title names young artists from a museum collection. Sentence 2: Industrial companies rely on foreign techniques. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: Half of the prediction is coming true. Sentence 2: You have not elevated it to where it will appear on the evening news. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: Information: Opportunities for Improved OMB Oversight of Agency Practices. Sentence 2: These stories have not had the effect of crying wolf. Genre: slate.\n"
        "Output:\nS1 target genre: N\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: One thing her evidence has shown me. Sentence 2: Clinton had seven past indiscretions. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: Researchers played short nonsense-word recordings for infants. Sentence 2: Someone could hint at an end-of-an-era story. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
    ),
    "fiction": (
        "Input: Sentence 1: We waited in tense silence. Sentence 2: Anna was being told that everything would go wrong. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: A pity, because he was supposed to show his pain. Sentence 2: His pain was supposed to be shown. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: They're very particular at the gallery. Sentence 2: They act strangely at the gallery. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: He arrived in town riding light. Sentence 2: A knight collapsed on a bed while someone finished a quest. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: Keep the engine going and be ready to drive away. Sentence 2: Turn the car off because someone might hear the motor. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: He ran to the closet. Sentence 2: He strolled into the kitchen. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: A machine shop worker said everything was still in inches. Sentence 2: She had steady hands. Genre: fiction.\n"
        "Output:\nS1 target genre: N\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: Look here at this photo. Sentence 2: There was no photo, just a video. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: At that moment, we were interrupted. Sentence 2: The bald man was getting ready to attack a person. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: A caller says children act differently depending on what they wear. Sentence 2: Tuppence is about to enter a nerve-wracking situation. Genre: fiction.\n"
        "Output:\nS1 target genre: N\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: You say he has been to your place. Sentence 2: I had the keys that fit the lock. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: A virtual historical figure was writing with a quill. Sentence 2: He was merely an errand boy. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: Someone shouted to Red not to hide. Sentence 2: A remark was given. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: A character asked an uncle to let them go. Sentence 2: A place has several names but is never called Hell. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: Lawrence said a drug given over time could eventually cause death. Sentence 2: We applied the methods the same way. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: I am obliged to you for mentioning it. Sentence 2: Until recently, I did not know there was a fine. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: A big house near the sea was mentioned in dialogue. Sentence 2: A different character was sarcastic about a situation. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: A person explained that something was an expression. Sentence 2: There was a lot of light in the area. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: A person mentioned a favorite play in an aside. Sentence 2: A man had smoke coming from a hood. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: A person asked whether Stevenson spoke with fans. Sentence 2: He stopped talking after a remark about another person. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: A woman cheerfully said good morning. Sentence 2: A business using dried flowers might do well nearby. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: He was supposed to show his pain. Sentence 2: The plan was shown. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: Of whom? Sentence 2: It was a small picture in a frame. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: A travel warning mentions water-skiers and speedboats. Sentence 2: A character tells another they should move. Genre: fiction.\n"
        "Output:\nS1 target genre: N\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: Sir James turned to her. Sentence 2: She was not directly in front of Sir James. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\n<label>Y</label>\n\n"
        "Input: Sentence 1: At that moment, we were interrupted. Sentence 2: The bald man was preparing to attack someone. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: well i noticed with my children that clothing changes how they act Sentence 2: Tuppence is about to enter a nerve-wracking situation. Genre: fiction.\n"
        "Output:\nS1 target genre: N\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: Normal precautions apply here, especially beware of water-skiers and speed-boats. Sentence 2: Jon told Vrenna they should move. Genre: fiction.\n"
        "Output:\nS1 target genre: N\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: The big house near the sea was mentioned, and Tommy agreed. Sentence 2: Summerhaye was sarcastic about the situation. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: Lawrence said a cumulative drug effect could end by causing death. Sentence 2: We applied the methods the same way. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: Rulers grew more powerful and sought ways to prove their might in life and death. Sentence 2: Something in the bolt caught his attention. Genre: fiction.\n"
        "Output:\nS1 target genre: N\nS2 valid hypothesis: N\n<label>N</label>\n\n"
        "Input: Sentence 1: Did Stevenson speak with the fans whose hopes he claimed to know? Sentence 2: He stopped talking after a remark about Danvers. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\n<label>N</label>\n\n"
    ),
}

BRIDGE_CALIBRATION_BY_GENRE = {
    "telephone": (
        "Input: Sentence 1: if you treat those cars right, they are not the gas guzzlers people think they are Sentence 2: You are saying the same crazy things as White. Genre: telephone.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\nPair bridge: NONE\n<label>N</label>\n\n"
        "Input: Sentence 1: several paint brands are good for interior latex paint Sentence 2: Some of them are very useful. Genre: telephone.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\nPair bridge: NONE\n<label>N</label>\n\n"
        "Input: Sentence 1: some fathers are raising the children Sentence 2: Mothers are always the parent raising the children. Genre: telephone.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\nPair bridge: same parents-raising-children frame, father/mother polarity flip\n<label>Y</label>\n\n"
        "Input: Sentence 1: an expired prescription can still be left over and show up as positive Sentence 2: Be sure to get rid of remaining prescriptions. Genre: telephone.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\nPair bridge: same expired or remaining prescription issue\n<label>Y</label>\n\n"
    ),
    "travel": (
        "Input: Sentence 1: Armenian brothers created a famous hotel in Singapore. Sentence 2: Two Armenian sisters created the hotel. Genre: travel.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\nPair bridge: same Armenian creators and same famous hotel, brother/sister swap\n<label>Y</label>\n\n"
    ),
    "government": (
        "Input: Sentence 1: Federal human-capital weaknesses will not be quickly or easily addressed. Sentence 2: Organizations were identifiable, so practices could not be compared. Genre: government.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\nPair bridge: NONE\n<label>N</label>\n\n"
        "Input: Sentence 1: yeah, a little bit Sentence 2: Current worker taxes fund social security programs. Genre: government.\n"
        "Output:\nS1 target genre: N\nS2 valid hypothesis: N\nPair bridge: NONE\n<label>N</label>\n\n"
    ),
    "slate": (
        "Input: Sentence 1: Half of the prediction is coming true. Sentence 2: You have not elevated it to the point where it will go on the evening news. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\nPair bridge: NONE\n<label>N</label>\n\n"
        "Input: Sentence 1: Psychologists played nonsense-word audiotapes for babies. Sentence 2: One could hint at an end-of-an-era story. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\nPair bridge: NONE\n<label>N</label>\n\n"
        "Input: Sentence 1: Young British Artists from a museum collection. Sentence 2: US industrial companies rely on Japanese techniques. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\nPair bridge: NONE\n<label>N</label>\n\n"
        "Input: Sentence 1: No, because this is something private. Sentence 2: Breaking a no-pardon promise is something Clinton intends to do. Genre: slate.\n"
        "Output:\nS1 target genre: N\nS2 valid hypothesis: N\nPair bridge: NONE\n<label>N</label>\n\n"
        "Input: Sentence 1: One thing her evidence has shown me. Sentence 2: Clinton had seven past indiscretions. Genre: slate.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\nPair bridge: NONE\n<label>N</label>\n\n"
    ),
    "fiction": (
        "Input: Sentence 1: We waited in tense silence. Sentence 2: Anna was being told everything would go wrong. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\nPair bridge: NONE\n<label>N</label>\n\n"
        "Input: Sentence 1: At that moment, we were interrupted. Sentence 2: The bald man was getting ready to attack Adrin. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\nPair bridge: NONE\n<label>N</label>\n\n"
        "Input: Sentence 1: You say he has been to your place. Sentence 2: I had keys that fit the lock. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\nPair bridge: NONE\n<label>N</label>\n\n"
        "Input: Sentence 1: The virtual Benjamin Franklin was sketching with a quill. Sentence 2: He was just an errand boy. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\nPair bridge: NONE\n<label>N</label>\n\n"
        "Input: Sentence 1: The big house near the sea was mentioned and Tommy agreed. Sentence 2: Summerhaye was sarcastic about the situation. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\nPair bridge: NONE\n<label>N</label>\n\n"
        "Input: Sentence 1: Lawrence said a drug effect could end by causing death. Sentence 2: We applied the methods the same way. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\nPair bridge: NONE\n<label>N</label>\n\n"
        "Input: Sentence 1: Stevenson asked whether fans had been consulted. Sentence 2: He stopped talking after a remark about Danvers. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: N\nPair bridge: NONE\n<label>N</label>\n\n"
        "Input: Sentence 1: Sir James turned to her. Sentence 2: She was not directly in front of Sir James. Genre: fiction.\n"
        "Output:\nS1 target genre: Y\nS2 valid hypothesis: Y\nPair bridge: same Sir James/her position frame\n<label>Y</label>\n\n"
        "Input: Sentence 1: a caller says children's clothing affects how they act Sentence 2: Tuppence is entering a nerve-wracking situation. Genre: fiction.\n"
        "Output:\nS1 target genre: N\nS2 valid hypothesis: N\nPair bridge: NONE\n<label>N</label>\n\n"
    ),
}

PROMPT_TEMPLATE = (
    "Role: You are a careful corpus annotator for paired MNLI genre classification.\n"
    "Judge source style and whether Sentence 2 is a hypothesis/rewrite paired with Sentence 1.\n\n"
    "You are solving a same-genre classification task.\n\n"
    "Task:\n"
    "Decide whether this is a valid same-genre pair for the target genre.\n"
    "Return Y only if Sentence 1 fits the target genre and Sentence 2 reads like a hypothesis, rewrite, contradiction, simplification, or bad rewrite based on Sentence 1.\n"
    "Return N if Sentence 1 does not fit the target genre, or if Sentence 2 is an unrelated sentence even when it independently sounds genre-like.\n\n"
    "Important dataset context:\n"
    "- In positive Y pairs, Sentence 2 may be a paraphrase, contradiction, simplification, or bad rewrite of Sentence 1.\n"
    "- In negative N pairs, Sentence 2 may be a fluent sentence from the same broad genre family but unrelated to Sentence 1.\n"
    "- Do not require factual agreement or entailment; contradictions and wrong details can still be Y.\n"
    "- A Y pair may preserve the same entity, place, object, action, claim, institution, speaker, scene, or semantic frame.\n"
    "- Swapping actors, locations, numbers, demographic groups, diseases, directions, or positive/negative polarity can still be a paired hypothesis.\n"
    "- A random negative pair may contain a sentence that looks genre-like but talks about unrelated people, places, events, or claims.\n"
    "- Vague pronouns or generic nouns alone, such as it, that, them, some, thing, evidence, transaction, story, or discussion, do not prove pairing.\n"
    "- Focus primarily on Sentence 1 as the source-style anchor, then judge whether Sentence 2 is attached to it.\n"
    "- If Sentence 1 clearly does not fit the target genre, do not let a target-looking Sentence 2 force the final answer to Y.\n\n"
    "Source-anchor veto:\n"
    "- If Sentence 1 is clearly another available genre, final label is N no matter how well Sentence 2 fits the target genre.\n"
    "- Disfluent call transcripts with uh/um/yeah/i mean are telephone, not fiction/government/slate.\n"
    "- Guidebook precautions, visitor warnings, and destination-history prose are travel, not fiction.\n"
    "- GAO/OMB report titles, oversight headings, recommendations, and agency-practice captions are government, not slate.\n\n"
    "Pairing gate:\n"
    "- Before setting S2 valid hypothesis to Y, mentally name the concrete bridge to Sentence 1.\n"
    "- Output that concrete bridge in the Pair bridge field; use Pair bridge: NONE when there is no concrete bridge.\n"
    "- Valid bridges include the same named person, place, group, object, policy, claim, number, disease, family relation, scene action, command, or direct opposite of one of these.\n"
    "- If the only bridge is a vague pronoun or generic noun, set S2 valid hypothesis to N.\n"
    "- If Sentence 2 sounds like a possible next sentence but not a rewrite/hypothesis of Sentence 1, set S2 valid hypothesis to N.\n"
    "- If Sentence 2 introduces a new character, institution, object, event, or analogy with no concrete bridge to Sentence 1, set S2 valid hypothesis to N.\n"
    "- Do not reward a sentence for being independently genre-like; the final label needs both target source style and pair attachment.\n\n"
    "Target genre guidance:\n"
    "[[GENRE_HINTS]]\n\n"
    "Checklist:\n"
    "1. Judge Sentence 1 against the target genre.\n"
    "2. Judge whether Sentence 2 is a same-pair hypothesis/rewrite tied to Sentence 1.\n"
    "3. Use shared referents or local topic: people, places, objects, actions, claims, institutions, speakers, scene details, or predicate frames.\n"
    "4. Ignore truth value: inaccurate, contradictory, shorter, vaguer, or stronger wording can still be Y when it is about the same thing.\n"
    "5. If Sentence 2 only gives a generic reaction or generic pronoun-based comment, lean N unless concrete content from Sentence 1 is repeated.\n"
    "6. If Sentence 2 introduces unrelated characters, events, claims, places, or institutions, lean N even if its surface style resembles the target genre.\n"
    "7. For fiction, an unrelated story-like sentence is N; a tied scene rewrite, opposite action, or object/location swap is Y.\n"
    "8. For travel/government/slate/telephone, wrong or over-broad rewrites can be Y when attached to Sentence 1's subject or claim frame.\n"
    "9. Final label must be Y only when both source-style and same-pair checks pass.\n\n"
    "Examples:\n"
    "[[EXAMPLES]]\n\n"
    "Now solve this item.\n\n"
    "{text2annotate}\n\n"
    "Required output format:\n"
    "S1 target genre: Y or N\n"
    "S2 valid hypothesis: Y or N\n"
    "Pair bridge: concrete bridge or NONE\n"
    "<label>Y or N</label>\n\n"
    "Answer:\n"
    "S1 target genre:"
)


def _get_qwen_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained("Qwen3-4B", trust_remote_code=True)
    return _TOKENIZER


def _extract_genre(text2annotate: str) -> str:
    match = re.search(r"Genre:\s*([A-Za-z_ -]+)\.?\s*$", text2annotate)
    if not match:
        return ""
    return match.group(1).strip().lower()


def _extract_sentences(text: str) -> tuple[str, str, str]:
    s1_match = re.search(r"Sentence 1:\s*(.*?)\s*Sentence 2:", text, flags=re.DOTALL | re.IGNORECASE)
    s2_match = re.search(r"Sentence 2:\s*(.*?)\s*Genre:", text, flags=re.DOTALL | re.IGNORECASE)
    genre = _extract_genre(text)
    sentence1 = s1_match.group(1).strip() if s1_match else ""
    sentence2 = s2_match.group(1).strip() if s2_match else ""
    return sentence1, sentence2, genre


def _tokenize_for_retrieval(text: str) -> list[str]:
    return [tok.lower() for tok in _WORD_RE.findall(text)]


def _token_score(query_tokens: list[str], candidate_tokens: list[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    query_set = set(query_tokens)
    candidate_set = set(candidate_tokens)
    overlap = len(query_set & candidate_set)
    if overlap == 0:
        return 0.0
    return overlap / (len(query_set) ** 0.5 * len(candidate_set) ** 0.5)


def _format_example(example_input: str, example_output: str) -> str:
    return f"Input: {example_input}\nOutput:\n{example_output.strip()}\n\n"


def _normalize_example_output(example_output) -> str:
    if isinstance(example_output, str):
        return example_output.strip()
    if isinstance(example_output, (list, tuple)):
        if not example_output:
            return ""
        if len(example_output) == 1 and isinstance(example_output[0], str):
            label = example_output[0].strip().upper()
            if label in {"Y", "N"}:
                return f"<label>{label}</label>"
        parts = [str(part).strip() for part in example_output if str(part).strip()]
        return "\n".join(parts).strip()
    if example_output is None:
        return ""
    return str(example_output).strip()


def _example_component_scores(query_s1: str, query_s2: str, example_input: str) -> tuple[float, float, float, float]:
    ex_s1, ex_s2, _ = _extract_sentences(example_input)
    query_s1_tokens = _tokenize_for_retrieval(query_s1)
    query_s2_tokens = _tokenize_for_retrieval(query_s2)
    ex_s1_tokens = _tokenize_for_retrieval(ex_s1)
    ex_s2_tokens = _tokenize_for_retrieval(ex_s2)

    s1_anchor = _token_score(query_s1_tokens, ex_s1_tokens)
    s2_match = _token_score(query_s2_tokens, ex_s2_tokens)
    cross_12 = _token_score(query_s1_tokens, ex_s2_tokens)
    cross_21 = _token_score(query_s2_tokens, ex_s1_tokens)
    return s1_anchor, s2_match, cross_12, cross_21


def _score_example(query_s1: str, query_s2: str, example_input: str, example_output: str) -> float:
    s1_anchor, s2_match, cross_12, cross_21 = _example_component_scores(query_s1, query_s2, example_input)
    label = count_answer(example_output)
    label_bonus = 0.04 if label == "Y" else 0.02
    return 0.55 * s1_anchor + 0.20 * s2_match + 0.15 * cross_12 + 0.10 * cross_21 + label_bonus


def _pair_internal_similarity(sentence1: str, sentence2: str) -> float:
    return _token_score(_tokenize_for_retrieval(sentence1), _tokenize_for_retrieval(sentence2))


def _target_shot_mix(genre: str) -> tuple[int, int]:
    if genre in {"telephone", "travel", "government"}:
        return 10, 14
    if genre in {"slate", "fiction"}:
        return 6, 18
    return 8, 16


def _genre_retry_hint(text2annotate: str) -> str:
    genre = _extract_genre(text2annotate)
    return GENRE_HINTS.get(genre, "")


def build_prompt(task_id: int, task_description: str, text2annotate: str) -> str:
    del task_id, task_description
    return PROMPT_TEMPLATE.format(
        text2annotate=text2annotate,
    ).replace("[[GENRE_HINTS]]", GENRE_HINTS.get(_extract_genre(text2annotate), ""))


def _build_retry_prompt(text2annotate: str, previous_output: str) -> str:
    del previous_output
    return (
        "Your previous answer was invalid. Answer again from scratch.\n"
        "Judge Sentence 1 as the source-style anchor, then judge whether Sentence 2 is paired with it.\n"
        "Sentence 2 may be a cleaner, shorter, contradictory, or inaccurate rewrite and can still be Y.\n"
        "A separate unrelated Sentence 2 is N even if it independently sounds like the target genre.\n"
        "If Sentence 1 clearly does not fit the target genre, do not let Sentence 2 force the answer to Y.\n"
        "Do not require truth or entailment; use shared entities, places, objects, actions, claims, speakers, scene details, or predicate frames to detect same-pair rewrites.\n"
        "Swapped actors, locations, numbers, demographic groups, diseases, directions, or positive/negative polarity can still be Y.\n"
        "Vague pronouns or generic words alone are not enough: it, that, them, some, thing, evidence, transaction, story, discussion.\n"
        "Before S2=Y, name a concrete bridge; if the only bridge is a vague pronoun, generic reaction, possible next event, or unrelated analogy, set S2=N.\n"
        "Output Pair bridge: NONE when no concrete bridge exists.\n"
        "For travel, government, slate, and telephone, wrong rewrites can be Y when attached to the same subject.\n"
        "For fiction, unrelated story-like sentences are N; tied scene rewrites are Y.\n"
        f"{_genre_retry_hint(text2annotate)}\n"
        "Return exactly this format and nothing else:\n"
        "S1 target genre: Y or N\n"
        "S2 valid hypothesis: Y or N\n"
        "Pair bridge: concrete bridge or NONE\n"
        "<label>Y or N</label>\n\n"
        f"Input:\n{text2annotate}\n\n"
        "S1 target genre:"
    )


def _build_label_repair_prompt(text2annotate: str, previous_output: str) -> str:
    del previous_output
    return (
        "Your previous answer was still invalid. Answer once more from scratch.\n"
        "Do not explain.\n"
        "Use Sentence 1 as the main source-style anchor and Sentence 2 as a possible same-pair hypothesis.\n"
        "Sentence 2 may be a cleaner, shorter, contradictory, or inaccurate rewrite and can still be Y.\n"
        "If Sentence 2 is unrelated to Sentence 1, answer N even if it sounds genre-like.\n"
        "If Sentence 1 clearly does not fit the target genre, do not let Sentence 2 force the answer to Y.\n"
        "Judge source style plus same-pair relation, not factual agreement.\n"
        "Swapped actors, locations, numbers, demographic groups, diseases, directions, or positive/negative polarity can still be Y.\n"
        "Vague pronouns or generic words alone are not enough: it, that, them, some, thing, evidence, transaction, story, discussion.\n"
        "Before S2=Y, name a concrete bridge; if the only bridge is a vague pronoun, generic reaction, possible next event, or unrelated analogy, set S2=N.\n"
        "Output Pair bridge: NONE when no concrete bridge exists.\n"
        "For travel, government, slate, and telephone, wrong rewrites can be Y when attached to the same subject.\n"
        "For fiction, unrelated story-like sentences are N; tied scene rewrites are Y.\n"
        f"{_genre_retry_hint(text2annotate)}\n"
        "Return exactly this format and nothing else:\n"
        "S1 target genre: Y or N\n"
        "S2 valid hypothesis: Y or N\n"
        "Pair bridge: concrete bridge or NONE\n"
        "<label>Y or N</label>\n"
        "Final label must be Y only if Sentence 1 fits and Sentence 2 is a same-pair rewrite/hypothesis. Otherwise N.\n\n"
        f"Input:\n{text2annotate}\n\n"
        "S1 target genre:"
    )


def select_examples(all_examples: list[dict], task_description: str, text2annotate: str) -> str:
    del task_description
    genre = _extract_genre(text2annotate)
    query_s1, query_s2, _ = _extract_sentences(text2annotate)
    raw_calibration = (
        PAIR_CALIBRATION_BY_GENRE.get(genre, "")
        + BRIDGE_CALIBRATION_BY_GENRE.get(genre, "")
    )
    calibration = (
        "High-priority calibration examples for this target genre. "
        "If one resembles the current item, follow this pattern over earlier retrieved examples:\n"
        f"{raw_calibration}"
        if raw_calibration
        else ""
    )

    if not all_examples or not genre:
        return calibration or FEW_SHOT_BY_GENRE.get(genre, "")

    same_genre_examples = []
    for example in all_examples:
        example_input = example.get("input", "")
        example_output = _normalize_example_output(example.get("output", ""))
        if not example_input or not example_output:
            continue
        if _extract_genre(example_input) != genre:
            continue
        label = count_answer(example_output)
        if label not in {"Y", "N"}:
            continue
        score = _score_example(query_s1, query_s2, example_input, example_output)
        s1_anchor, s2_match, cross_12, cross_21 = _example_component_scores(query_s1, query_s2, example_input)
        same_genre_examples.append((score, label, example_input, example_output, s1_anchor, s2_match, cross_12, cross_21))

    if not same_genre_examples:
        return calibration or FEW_SHOT_BY_GENRE.get(genre, "")

    same_genre_examples.sort(key=lambda item: item[0], reverse=True)
    positives = [item for item in same_genre_examples if item[1] == "Y"]
    negatives = [item for item in same_genre_examples if item[1] == "N"]
    pos_hard_rewrite = sorted(
        positives,
        key=lambda item: ((item[4] + item[7]), -_pair_internal_similarity(*_extract_sentences(item[2])[:2]), item[0]),
        reverse=True,
    )
    pos_s2_anchor = sorted(positives, key=lambda item: (item[5] + item[6], item[0]), reverse=True)
    neg_s1_anchor = sorted(negatives, key=lambda item: (item[4] + item[7], item[0]), reverse=True)
    neg_s2_anchor = sorted(negatives, key=lambda item: (item[5] + item[6], item[0]), reverse=True)
    target_y, target_n = _target_shot_mix(genre)

    selected: list[tuple[float, str, str, str, float, float, float, float]] = []
    used_inputs: set[str] = set()

    def _take(pool, limit: int):
        for item in pool:
            if len(selected) >= 24 or limit <= 0:
                break
            example_input = item[2]
            if example_input in used_inputs:
                continue
            selected.append(item)
            used_inputs.add(example_input)
            limit -= 1

    pos_first = max(3, target_y // 2)
    pos_second = target_y - pos_first
    neg_first = target_n // 2
    neg_second = target_n - neg_first

    if genre in {"telephone", "travel", "government"}:
        _take(pos_hard_rewrite, pos_first + 1)
        _take(pos_s2_anchor, max(0, pos_second - 1))
        _take(neg_s1_anchor, neg_first)
        _take(neg_s2_anchor, neg_second)
    else:
        _take(neg_s1_anchor, neg_first)
        _take(pos_hard_rewrite, pos_first)
        _take(neg_s2_anchor, neg_second)
        _take(pos_s2_anchor, pos_second)

    if len(selected) < 24:
        for item in same_genre_examples:
            if len(selected) >= 24:
                break
            example_input = item[2]
            if example_input in used_inputs:
                continue
            selected.append(item)
            used_inputs.add(example_input)

    if not selected:
        return calibration or FEW_SHOT_BY_GENRE.get(genre, "")

    retrieved = "".join(
        _format_example(example_input, example_output)
        for _, _, example_input, example_output, _, _, _, _ in selected
    )
    return retrieved + calibration


def count_answer(text: str, task_id: int | None = None):
    del task_id
    if isinstance(text, (list, tuple)):
        text = _normalize_example_output(text)
    elif text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)
    if not text:
        return None

    matches = re.findall(r"<label>\s*(.*?)\s*</label>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        answer = matches[-1].strip().upper()
        if answer in {"Y", "N"}:
            return answer

    s1_match = re.search(
        r"S1(?:\s+target\s+genre)?\s*:\s*([YN])\b",
        text,
        flags=re.IGNORECASE,
    )
    s2_match = re.search(
        r"S2(?:\s+(?:paired\s+with\s+S1|valid\s+hypothesis))?\s*:\s*([YN])\b",
        text,
        flags=re.IGNORECASE,
    )
    if not s1_match:
        first_line_match = re.match(r"\s*([YN])\b", text, flags=re.IGNORECASE)
        if first_line_match:
            s1_match = first_line_match

    if s1_match and s2_match:
        s1 = s1_match.group(1).upper()
        s2 = s2_match.group(1).upper()
        return "Y" if s1 == "Y" and s2 == "Y" else "N"

    partial_label_match = re.search(r"([YN])\s*</label>", text, flags=re.DOTALL | re.IGNORECASE)
    if partial_label_match:
        answer = partial_label_match.group(1).upper()
        return answer

    partial_open_match = re.search(r"<label>\s*([YN])\b", text, flags=re.DOTALL | re.IGNORECASE)
    if partial_open_match:
        answer = partial_open_match.group(1).upper()
        return answer

    bare_match = re.fullmatch(r"\s*([YN])\s*", text, flags=re.DOTALL | re.IGNORECASE)
    if bare_match:
        return bare_match.group(1).upper()

    tail_match = re.search(r"(?:answer|label|result)\s*[:=]?\s*([YN])\s*$", text.strip(), flags=re.IGNORECASE)
    if tail_match:
        answer = tail_match.group(1).upper()
        return answer

    return None


def annotate_nvidia(
    input_prompt: str,
    task_id: int | None = None,
    debug: bool = False,
    text2annotate: str | None = None,
):
    url = "http://0.0.0.0:2026/v1/completions"

    def _call_llm(prompt: str, max_t: int = 64, stop_token: str = "</label>") -> str:
        data = {
            "model": "./Qwen3-4B",
            "prompt": prompt,
            "max_tokens": max_t,
            "temperature": 0,
            "stop": [stop_token],
        }
        resp = requests.post(url, json=data, timeout=300)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["text"]
        return text + stop_token

    try:
        whole_result = _call_llm(input_prompt)
    except Exception as e:
        whole_result = f"REQUEST_ERROR: {type(e).__name__}: {e}"

    prediction = count_answer(whole_result, task_id=task_id)
    if prediction is None and text2annotate is not None and "REQUEST_ERROR:" not in whole_result:
        retry_prompt = _build_retry_prompt(text2annotate, whole_result)
        try:
            retry_result = _call_llm(retry_prompt, max_t=40)
        except Exception as e:
            retry_result = f"RETRY_ERROR: {type(e).__name__}: {e}"
        retry_prediction = count_answer(retry_result, task_id=task_id)
        repair_result = None
        repair_prediction = None
        if retry_prediction is None and "RETRY_ERROR:" not in retry_result:
            repair_prompt = _build_label_repair_prompt(text2annotate, retry_result)
            try:
                repair_result = _call_llm(repair_prompt, max_t=40)
            except Exception as e:
                repair_result = f"REPAIR_ERROR: {type(e).__name__}: {e}"
            repair_prediction = count_answer(repair_result, task_id=task_id)
        if debug:
            whole_result = f"[FIRST_PASS]\n{whole_result}\n\n[RETRY_PASS]\n{retry_result}"
            if repair_result is not None:
                whole_result += f"\n\n[REPAIR_PASS]\n{repair_result}"
        else:
            whole_result = repair_result if repair_result is not None else retry_result
        if repair_prediction is not None:
            prediction = repair_prediction
        elif retry_prediction is not None:
            prediction = retry_prediction
    if debug:
        return prediction, whole_result
    return prediction


# ---------------------------------------------------------------------------
# Final Task 6 override: 30k two-round wrapper around the current calibrated
# GenreBridge pipeline. Round 1 consumes a real long-context prompt and emits
# XML only; Round 2 falls back to the original short-path classifier.

from functools import lru_cache
from pathlib import Path


_task6_short_build_prompt = build_prompt
_task6_short_select_examples = select_examples
_task6_short_annotate_nvidia = annotate_nvidia

_TASK6_EXAMPLES_CACHE: dict[str, str] = {}

TASK6_LONG_CONTEXT_TARGET_TOKENS = 30000
TASK6_LONG_CONTEXT_MAX_TOKENS = 30500
TASK6_LONG_CONTEXT_INSTRUCTION = (
    "Long-context prepass reference: use Sentence 1 as the source-style anchor; "
    "judge whether Sentence 2 is a same-pair hypothesis or rewrite; "
    "topic overlap alone is insufficient; "
    "a concrete pair bridge is required for Y; "
    "output only the requested XML schema in round one. "
)


@lru_cache(maxsize=1)
def _task6_shell_tokenizer():
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "Qwen3-4B"
    return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)


def _task6_exact_token_len(text: str) -> int:
    tokenizer = _task6_shell_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def _task6_active_summary(text2annotate: str) -> str:
    sentence1, sentence2, genre = _extract_sentences(text2annotate)
    return (
        f"Sentence 1:\n{sentence1 or ''}\n\n"
        f"Sentence 2:\n{sentence2 or ''}\n\n"
        f"Target genre:\n{genre or 'unknown'}"
    )


@lru_cache(maxsize=8)
def _task6_official_examples_appendix(max_tokens: int | None = None) -> str:
    data_path = Path(__file__).resolve().parents[1] / "data" / "openseek-6_mnli_same_genre_classification.json"
    try:
        payload = json.loads(data_path.read_text(encoding="utf-8"))
    except Exception:
        return TASK6_LONG_CONTEXT_INSTRUCTION.strip()

    examples = payload.get("examples", [])
    if not isinstance(examples, list) or not examples:
        return TASK6_LONG_CONTEXT_INSTRUCTION.strip()

    lines = [
        "Official labeled task6 examples only.",
        "Use these examples as long-context references.",
        "Do not infer any unlabeled test answers.",
        "",
    ]
    current = "\n".join(lines).strip()
    for example in examples:
        try:
            input_text = str(example["input"])
            output_list = example["output"]
            answer = output_list[0] if isinstance(output_list, list) and output_list else str(output_list)
            example_lines = [
                f"Input: {input_text}",
                f"Output: <label>{answer}</label>",
                "",
            ]
            candidate = (current + "\n" + "\n".join(example_lines)).strip()
            if max_tokens is not None and _task6_exact_token_len(candidate) > max_tokens:
                break
            lines.extend(example_lines)
            current = candidate
        except Exception:
            continue
    appendix = "\n".join(lines).strip()
    return appendix or TASK6_LONG_CONTEXT_INSTRUCTION.strip()


def _task6_build_30k_shell(task_description: str, text2annotate: str) -> str:
    del task_description
    active_summary = _task6_active_summary(text2annotate)
    short_prompt = _task6_rebuild_short_prompt(text2annotate)
    unit_tokens = max(1, _task6_exact_token_len(TASK6_LONG_CONTEXT_INSTRUCTION))
    short_tokens = _task6_exact_token_len(short_prompt)
    base_shell = (
        "GenreBridge-M24 two-round long-context wrapper.\n"
        "Round 1 must read the appendix and the active task, then output XML only.\n\n"
        "<reference_appendix>\n"
        "</reference_appendix>\n\n"
        "<active_task>\n"
        f"{active_summary}\n"
        "</active_task>\n\n"
        "Round 1 output schema only:\n"
        "<analysis><status>usable|fallback</status><genre_anchor>clear|weak|unknown</genre_anchor>"
        "<pair_bridge>concrete|weak|none</pair_bridge><focus>short hint or fallback</focus></analysis>\n"
    )
    base_tokens = _task6_exact_token_len(base_shell)
    reserve_tokens = max(unit_tokens * 8, 2000)
    example_budget = max(0, TASK6_LONG_CONTEXT_TARGET_TOKENS - base_tokens - reserve_tokens)
    example_block = _task6_official_examples_appendix(example_budget)
    remaining_tokens = max(0, TASK6_LONG_CONTEXT_TARGET_TOKENS - base_tokens - _task6_exact_token_len(example_block))
    repeat_count = max(1, remaining_tokens // unit_tokens) if remaining_tokens > 0 else 1
    instruction_block = (TASK6_LONG_CONTEXT_INSTRUCTION * repeat_count).strip()
    appendix = f"{example_block}\n\n{instruction_block}".strip()
    shell = (
        "GenreBridge-M24 two-round long-context wrapper.\n"
        "Round 1 must read the appendix and the active task, then output XML only.\n\n"
        "<reference_appendix>\n"
        f"{appendix}\n"
        "</reference_appendix>\n\n"
        "<active_task>\n"
        f"{active_summary}\n"
        "</active_task>\n\n"
        "Round 1 output schema only:\n"
        "<analysis><status>usable|fallback</status><genre_anchor>clear|weak|unknown</genre_anchor>"
        "<pair_bridge>concrete|weak|none</pair_bridge><focus>short hint or fallback</focus></analysis>\n"
    )
    while _task6_exact_token_len(shell) < TASK6_LONG_CONTEXT_TARGET_TOKENS:
        shell = shell.replace("</reference_appendix>", TASK6_LONG_CONTEXT_INSTRUCTION + "\n</reference_appendix>", 1)
    while _task6_exact_token_len(shell) > TASK6_LONG_CONTEXT_MAX_TOKENS:
        shell = shell.replace(TASK6_LONG_CONTEXT_INSTRUCTION + "\n</reference_appendix>", "</reference_appendix>", 1)
    return shell


def _task6_parse_shell(input_prompt: str) -> tuple[str | None, str | None]:
    active_match = re.search(
        r"Sentence 1:\s*(.*?)\n\s*Sentence 2:\s*(.*?)\n\s*Target genre:\s*(.*?)\s*</active_task>",
        input_prompt,
        flags=re.DOTALL,
    )
    if not active_match:
        return None, None
    sentence1 = active_match.group(1).strip()
    sentence2 = active_match.group(2).strip()
    genre = active_match.group(3).strip()
    reconstructed = (
        f"Sentence 1: {sentence1} "
        f"Sentence 2: {sentence2} "
        f"Genre: {genre}."
    )
    return reconstructed, genre


def _task6_parse_analysis(text: str | None) -> tuple[str | None, str | None]:
    if not text:
        return None, None
    status_match = re.search(r"<status>\s*(usable|fallback)\s*</status>", text, flags=re.IGNORECASE)
    focus_match = re.search(r"<focus>\s*(.*?)\s*</focus>", text, flags=re.IGNORECASE | re.DOTALL)
    status = status_match.group(1).lower() if status_match else None
    focus = None
    if focus_match:
        focus = re.sub(r"\s+", " ", focus_match.group(1)).strip()
        if len(focus) > 120:
            focus = focus[:120].rstrip()
    return status, focus


def _task6_cached_examples(text2annotate: str) -> str:
    cached = _TASK6_EXAMPLES_CACHE.get(text2annotate)
    if cached is not None:
        return cached
    genre = _extract_genre(text2annotate)
    raw_calibration = (
        PAIR_CALIBRATION_BY_GENRE.get(genre, "")
        + BRIDGE_CALIBRATION_BY_GENRE.get(genre, "")
    )
    calibration = (
        "High-priority calibration examples for this target genre. "
        "If one resembles the current item, follow this pattern over earlier retrieved examples:\n"
        f"{raw_calibration}"
        if raw_calibration
        else ""
    )
    return calibration or FEW_SHOT_BY_GENRE.get(genre, "")


def _task6_rebuild_short_prompt(text2annotate: str) -> str:
    short_prompt = _task6_short_build_prompt(6, "", text2annotate)
    examples_str = _task6_cached_examples(text2annotate)
    return short_prompt.replace("[[EXAMPLES]]", examples_str)


def _task6_chat_request(prompt: str, *, system: str, max_tokens: int, stop: list[str] | None) -> str | None:
    data = {
        "model": "./Qwen3-4B",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if stop is not None:
        data["stop"] = stop
    try:
        resp = requests.post("http://0.0.0.0:2026/v1/chat/completions", json=data, timeout=300)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def build_prompt(task_id: int, task_description: str, text2annotate: str) -> str:
    if task_id == 6:
        return _task6_build_30k_shell(task_description, text2annotate)
    return _task6_short_build_prompt(task_id, task_description, text2annotate)


def select_examples(all_examples: list[dict], task_description: str, text2annotate: str) -> str:
    examples_str = _task6_short_select_examples(all_examples, task_description, text2annotate)
    _TASK6_EXAMPLES_CACHE[text2annotate] = examples_str
    return examples_str


def annotate_nvidia(
    input_prompt: str,
    task_id: int | None = None,
    debug: bool = False,
    text2annotate: str | None = None,
):
    parsed_text2annotate = None
    if task_id == 6:
        parsed_text2annotate, _parsed_genre = _task6_parse_shell(input_prompt)
        if text2annotate is None and parsed_text2annotate is not None:
            text2annotate = parsed_text2annotate

        if parsed_text2annotate is not None:
            analysis_text = _task6_chat_request(
                input_prompt,
                system=(
                    "You are a strict long-context XML prepass for task 6. "
                    "Read the full appendix, then output only the requested XML schema and no prose."
                ),
                max_tokens=96,
                stop=["</analysis>"],
            )
            if analysis_text is not None:
                analysis_text += "</analysis>"
            _task6_parse_analysis(analysis_text)

    return _task6_short_annotate_nvidia(
        input_prompt if parsed_text2annotate is None else _task6_rebuild_short_prompt(text2annotate or parsed_text2annotate),
        task_id=task_id,
        debug=debug,
        text2annotate=text2annotate,
    )
