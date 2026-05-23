import json
import os
import time
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def task5_data_loader(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("task_id"), data.get("examples", []), data.get("test_samples", [])


client = OpenAI(
    api_key="dummy",  # 你的接口如果不需要密钥，填任意字符串即可
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
            time.sleep(2) 
# 84%版本
# sentiment_prompt = """请判断下面句子的感情，作者是否感到悲伤
# 【重要准则】：
# 在这个任务中，“悲伤”是一个广义概念。如果句子表达了以下任何一种情绪，请均判断为“sad”：
# 1. 失望或期待落空（如：对服务不满、对某人表现失望）。
# 2. 委屈、被威慑或无力感（如：intimidated, pout, 被人羞辱/dragged）。
# 3. 极度愤怒引发的沮丧（如：fuming, raging）。
# 4. 遗憾、孤独或心碎。
# 5. 在表情和文字冲突的情况下，以文字表达的情绪为准（如：虽然表情是哭泣，但文字表达的是羡慕、震惊等）。
# 请无视推特中的用户名（如@user）和链接，如果句子完全中性，或者表达的是纯粹的快乐、兴奋，则判断为“not sad”。
# 直接给出“sad”或者“not sad”。
# """
import re

def clean_text(text):
    # 匹配 @ 开头，后面跟着字母数字或下划线的字符，并将其替换为空空格
    # \w 等价于 [a-zA-Z0-9_]
    cleaned_text = re.sub(r'@\w+', '', text)
    
    # 进一步优化：去除因删掉 @ 导致的重复空格
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

FEW_SHOT = """
输入：@badpostyoongi I know for a fact they'll either ignore the fact tiff isn't or change Cindy's background
输出：not sad

输入：Went to bed a 1:30, fell asleep after, my niece started crying at 4. I'm dying... 😧
输出：sad

输入：and i shouldve cut them off the moment i started hurting myself over them :o
输出：sad

输入：@bxchpls03 U so lucky ahu 😭
输出：not sad

输入：@Jack_Septic_Eye Grass growing simulator is offended
输出：sad

输入：@RealSkipBayless Your opinions on sports is dreadful
输出：sad

输入：@NewsByKatherine @jonkarl @ABC YOU GUY'S SOUND astounded!! Does anyone working w Trump WH have an any ethical,moral, values? 🙈🙉🙊
输出：not sad

输入：Do people notice that only saying 'You're so pretty' when I have make-up on. Is offense! \n& I take note that they never say it when I don't.
输出：not sad

输入：Alaina and I are at 90 days on our snap streak. So?
输出：not sad

输入：United Airline at Newark needs more Kiosks so that people won't miss their cut off time.  And hire more ppl too.  #horrible
输出：sad

输入：If @TheRock's Presidential run is as bad as his appearance in #Baywatch neither party need fear his run. That's $8 I'll never get back #crap
输出：sad

输入：@chelseanews4you Best of luck blues army make us feeling goofy this season again  BLUES TILL ETERNITY
输出：not sad

输入：@praddy06 sir.. will we have the need for umbrella today evening.. sun seems to be stronger to pave way for clouds.. 😟
输出：sad

输入：@washingtonpost @silvajanes How awful!!!!!!
输出：sad

输入：@_Oteraw unhappy and unfulfilled 😂
输出：sad

输入：When I think about Yondu & his crew, Rocket, & Groot doing 700 jumps to Ego planet, I start laughing.
输出：not sad

输入：Same ☹
输出：sad

输入：@BillMoyersHQ Elderly belong w/family; our social system is broken.  This egregious 'Meancare' will shake it up, alas.
输出：sad

输入：I have actually watch drugs destroy an entire family 😢Mother's on skid row. Oldest daughter lost her child. Father is estranged. #horrific
输出：sad

输入：I feel intimidated
输出：sad

输入：@CTAFails @cta is it ok for your drivers to smile not open the door and drive off 
输出：sad

输入：@RyanHedderick happy birthday big fella, have a good one. up the blues x
输出：not sad

输入：@maidinaustralia D: That's horrid. *hugs*
输出：sad

输入：#Rage and #disappointment man.... Is that life or what? Lol
输出：sad

输入：So disappointed in E! portrayal of Kylie Jenner! Makes her look so fake &amp; filthy rich @ every turn #disappointment @enews #LifeofKylie 🤢😒
输出：sad

输入：WAIT...Lawrence's friend dragged the fuck outta him!! 
输出：sad

输入：Hiya everyone if you want please #retweet my pin #rt #help #romance #wattpad    #hurt #tweet #twitter #thanks
输出：not sad

输入：Can't handle rude people. Doesn't matter what job you do, a consultant or not, treat people how you would like to be treated 😡 #disapointed
输出：sad

输入：Are we making the dark, darker or are we shining the light of Jesus into the dark.\n #Jesus  #light #shine
输出：not sad

输入：Worst dreams. 😥
输出：sad

输入：@apinknumjoo Hello, namjoo unnie! Welcome to paradox.💕 i'm deeply sorry for the late greeting 😥 Chaey is wishing you to have a pleasant +
输出：not sad

输入：#IfOnlyPeopleWould not exist. #humanity #life #ignorance #nature #mothernature #sad #disappointment #smh #personal #opinion #views #animals
输出：sad

输入：@1720maryknoll I was #fuming Kenny.
输出：sad

输入：+++ '#Dearly #beloved, avenge not yourselves, but rather give place unto #wrath: for it is #written, #Vengeance is #mine; I …' #Romans12v19
输出：not sad

输入：@tonyposnanski Putin told him not to. #pout #heylookatthedistraction
输出：sad

输入：@brownjayson @MylesGorham85 I subscribe to this same philosophy but then drafted Michael Floyd anyways because I'm #bad at this
输出：sad

输入：@_Buddh_ @rohit_mhpl @indujalali @rajnathsingh We are also wating when terrorism willb history. And one thing kashmir is not India's
输出：not sad

输入：Something #awestruck me today as i was laughing to a non subtitle korean variety show...
输出：not sad

输入：Selling nudes pics and vids kik me to buy! Dirty_becca69\n\n#kik #kikme #kikusernames #snap #snapchat #findom #nudes #slut #kiktrade #horny
输出：not sad

输入：@Zineeta_R @RusEmbUSA @mfa_russia The hatred and fear many russians have for anything non-russian is just sad.
输出：sad

输入：@AGlyndwr @TeaPainUSA All I know is the sentence will start with 'look...' like any high school punk would start a threat.
输出：not sad

输入：All and boy play n0 no play dull and mᴬkes.
输出：not sad

输入：Wow! Today, totally seeing alot of #mean #people out in this #world! Turn it around and #start cultivating #kindness! #SuccessTRAIN #warrior
输出：not sad

输入：It’s lack of #faith that makes #people #afraid of #meeting #challenges …\n\n#MuhammadAli
输出：sad

输入：Can't believe Zain starting secondary this year 😢
输出：sad

输入：When you have just about enough @marmite in your jar at work for 1/4 of a slice of toast 😩😩 #unhappy
输出：sad

输入：@GarfieldLineker @TimCRoberts *on his back. Apologies for retweeting a tweet with grammatical error #mybad 😱
输出：sad

输入：Literally hanging on by a thread need some taylor ray tonight loving a bad dog sucks #taylorrayholbrook #hurting @TaylorRaysTweet
输出：sad

输入：@skh4808 @theveteran425FA @TomiLahren Then why'd they wait until now to start getting pissy?
输出：not sad

输入：When the lights shut off and it's my turn to settle down, my main concern...
输出：not sad

输入：@theIeansquad @SatanHeavenly Rap is so unbearable and horrific.
输出：sad

输入：@imaorangepeeler Imagine you walking up them sober 😉
输出：not sad

输入：Had frustration dream that left me utterly f**king furious. Plus side: so angry couldn't sleep, wrote 1500 words. Minus side: still raging!
输出：sad

输入：There's no excuse for making the same mistakes twice. Live & Learn or deal with the consequences of being unhappy #truthbomb
输出：sad

输入：I am shy at first.It usually takes me a few minutes to assess the jaw of the people i am hanging out with and then i will act according🤷🏽‍♀️
输出：not sad

输入：5 goals in 87 appearances last season between McKay, Holt and Windass! Simply is horrific! Get midfield balance sorted and team will fire!
输出：sad

输入：@NitashaKaul @Snehakaul2Kaul so beautiful dear, thanks,everybody knows it is in benefit of India & GOI has done this terror attack as before
输出：not sad

输入：i just wanna be sober with u
输出：not sad

输入：'we need to do something. something must be done!!!!!'\n\nyour anxiety is amusing. nothing will be done. despair.
输出：sad

输入：@TheView Joy isn't a comedian. She's a bully for fat shaming the governor. Great example she's setting for her grandson.
输出：sad

输入：[ @TheChicMystique ] — hurting badly and that he can't just leave him like that. Angry and heartless. ]\n\nI promise you that I'll be back, —
输出：sad

输入：I've got builders in my office and I have a game to make. Perhaps ill start sketching the next game... #gamedev #indiedev
输出：not sad

输入：@MikeAndMike @Buster_ESPN if you get time.. @Orioles buyers it sellers or what are we going to do!!?! #panic
输出：sad

输入：I like the #glow in the #dark #fidgetspinner. Not because it glows the dark neither. It feels more lighter and smoother than the others
输出：not sad

输入：Damn I lost my keys and I forgot to get the garage opener
输出：sad

输入：Shame the cashback @mbna @AmexUK credit card comes to an end. I used to look forward to that end of year bonus. Sad really. #cashback
输出：sad

输入：@JoyceMeyer @mrsglessman #day of #vengeance of our #God; To #comfort all who #mourn, To console those who #mourn in #Zion, To give [4/7]
输出：not sad

输入：When Duane Allman died, I learned to appreciate Stevie Ray Vaughan. True story.  #blues #legends
输出：not sad

输入：made up my mind to \nmake a new start
输出：not sad

输入：nomore drinking for me 😌😂 #serious
输出：not sad

输入：We can replace #loss with #hope, #hate with #love, #pain with #gain, if we close the window of #bitterness and open doors of #faith. #TryIt
输出：not sad

输入：ARMYs we see you working hard to keep BTS on the Social 50 chart!\nDon't feel discouraged, it's still amazing that BTS is #2 with no promos 💕
输出：not sad

输入：Okay I seriously don't know how this whole twitter thing works #lost
输出：not sad

输入：@andyfleming83 Bastard squirrels. 😡
输出：not sad

输入：#LouiseLinton - haters gonna hate keep on being your#fabulous self they'll keep on being  #miserable
输出：sad

输入：Please stop ruining my depressing memes with your positivity and optimism
输出：sad

输入：#Worry never robs tomorrow of its #sorrow; it only saps today of its #strength - A. J. Crown #faith #positive #motivation
输出：not sad

输入：What does Amelia want?! Sarah was v grateful  #CBB
输出：not sad

输入：If you sit back, watch & listen to every .@TheDemocrats & @DNC member, you'll quickly learn it's #Victimhood, #racism, & #hatred. I'm #WOKE
输出：sad

输入：I mean, not that I wanted goats to faint... but I wanted to see the goats faint.  #eclipse
输出：not sad

输入：Really don't want my mom to go back home 😢 😢 😢 😢 😢 😢 😢  #gutted #crying #miserable #why
输出：sad

输入：Was a huge fan of @Ryanair but last few flights have been horrific. #rude #poorservice #nostock etc etc etc #dissapointed
输出：sad

输入：Africa has unique and tremendous problems with war, overpopulation, starvation and tribalism which makes moving Africa forward hard.
输出：sad

输入：Do not presume that richness of poorness will bring you happiness - Santosh Kalwar #quote #mentalhealth #psychology #depression #anxiety
输出：sad

输入：coffee the floor; And the soul grew furious as the stillness broken by little,
输出：sad

输入：@jaassiieeeee I will not fall to the dark side😂😂
输出：not sad

输入：@emmajckson awe thank you (,:
输出：not sad

输入：@BBCBreaking I have some mistrust of the medical profession.  The cover up was more important than the patients.
输出：sad

输入：Look upon mine #affliction & my ​​​#pain​; & forgive all my sins. -Ps 25:18
输出：sad

输入：Caleb had a nightmare about zombies. I had a dream about freedom.......
输出：not sad

输入：The next time I go to Lagos I will gate crash somebody's owambe dressed in lace and gele to eat amala and shake my waist😑
输出：not sad

输入：Damn I'm tired as hell I never get a off day during the week anymore 😭 I wanna call in so bad but these lil 60 hrs sounds so good.
输出：sad

输入：A bih be on lock down and shit. #depressing
输出：sad

输入：How do you feel about @Snapchat new feature #SnapMap 👎👍❓ #twitterpoll #polls #vote #Poll #Snapchat  #twitter #tech #technology
输出：not sad

输入：So now I have to buy a whole new computer 
输出：not sad

输入：@who_cares_nvm The destroying of my memory is my goal so ECT might work. Or a zapper thingy like in Men in Black. Or Dumbledore's pensive.
输出：not sad

输入：@mrjamesob @LBC 😂 snowflake random such a funny man never a dull moment brilliant
输出：not sad

输入：Good morning and happy Tuesday! I hope you have a terrific day! Enjoy it tons 😃
输出：not sad

输入：I've come to the conclusion that the online world is seriously 'fucked up' there's absolutely no other words to describe #bleak  #grim
输出：sad

输入：Imagine suffering chronic depression  and being told 'you have an unattractive chip on your shoulder' #DWP #WRAG #WWW.GOV.UK #Mentalhealth
输出：sad

输入：@realDonaldTrump You've spent more time and energy protecting Michael Flynn than your own son. You're a coward and an awful parent.
输出：sad

输入：@thealexpeace Not bad at All, think may come back to haunt you's
输出：not sad

输入：@UNESCO Remember not to spread #hatred and #fakenews on the internet. Do not abuse #Hashtag10 for spreading #ArabNationalism!
输出：sad

输入：Leviticus 19:14\nYou shall not curse the #deaf or put a stumbling #block before the #blind, but you shall #fear your #God: I am the [1/2]
输出：not sad

输入：I never thought I would say this but I really miss Todd 😥
输出：sad

输入：Can't Talk To An Incompetent Person. Goes In One Ear and Out The Other. #irritated #NoPoint #MassiveEyeRoll
输出：sad

输入：was one moron  driving his oversize tonka truck with the big flag in the bed back and forth blaring country music. 😐 #disappointment
输出：sad

输入：i don't understand ppl who save wasps , next chance that lil dude gets he gnna sting ur grandma
输出：not sad

输入：Counting on you, Queensland. #StateOfOrigin #Broncos #maroons #blues #NSWBlues #qld
输出：not sad

输入：@JusticeWillett Lord, we don't understand tragedy. Do what You do best: bring good grom it and comfort those who mourn. Amen
输出：sad

输入：All this makeup is going on sale....\nBut I ain't got the funds. #heartbreaking
输出：sad

输入：@uzalu_ @Veeh_Ro What a joyless cunt.
输出：sad

输入：@BrettKeeble I guess it never. @smartassunit #worry
输出：not sad

输入：#World do you know what the difference is between #drunk and #sober? One word. #Coherence.
输出：sad

输入：EEEEEKKKK!!!!\nProduct LAUNCH 😍✋💖\nI'm am literally B•U•Z•Z•I•N•G!!!\nSingle sachets 😍😍\n \nMessage me for yours! 😜🙆💜#loveyourlifestyle #shakes
输出：not sad

输入：I posted a snap of my dad, and someone thought he was my GRANDMOTHER #crying
输出：not sad

输入：Last Sunday YouTube glitch making me lose up to 20 subs is heartbreaking for a channel my size! Nearly at 1K though #1Ksubs #youtube soon 😀
输出：not sad

输入：Nah but as a governor how do you call someone a bum & that you love calls from 'communist in Montclair' ? 😂 #crying
输出：not sad

输入：Too much caffeine, and I'm dyyyying. #jitters #shakes #paranoia #heartbeat1000 ☕💔
输出：sad

输入：@KitchenAidUSA  I spent over $500 on your mixer, yet the dough hook chips in my dough.  I buy a new one and the same thing happens. #unhappy
输出：sad

输入：Post TRNSMT blues
输出：not sad

输入：@Bravotv is there a way to watch NYC Million Dollar Listing & filter @FredrikEklundNY OUT of the episodes? #primadonna #duckface #tantrum
输出：sad

输入：@SkyUK not impressed by your customer support. Forcing customers to use fb chat or sms! Very slow. issue is not getting sorted
输出：sad

输入：Beware the wrath of an angry, frustrated, #agile grandma with a network. 👵🏼😡 I'm just sayin'. #objectlesson
输出：sad

输入：@Cmdr_Hadfield CNN's Wolf Blitzer calls you an American astronaut and you don't correct him? #dissapointed
输出：sad

输入：Did men call themselves shy and mean it? So I reassure him that I'm just making sure he's a good investment and alla that 🙄
输出：not sad

输入：She was obviously moved by the music of @RobertCrayBand tonight and wanted to share the love. #blues #concert
输出：not sad

输入：Girls masturbate too,boys cry too!\n#girls # boys #cry #masturbate
输出：not sad

输入：@angrydwarf9 @carolinesandall It ruins my frigging night each night at 9pm. Mrs loves it, i've been early to bed for a month. #dreadful
输出：sad

输入：@RoflCritic @NBTDilli @SudamaNBT ,BJP MCD busy collection of suvidha sulk from unauthorised colonies
输出：not sad

输入：And I'm really pissed the fuck off because I do a good job of keeping my kid well because I don't like to see her sick and sad.
输出：sad

输入：I like how all itos manga end with the most bleak and hopeless endings, but not doing it in a way to make it look like the protagonist lost
输出：sad

输入：@ScottAdamsSays broke it down on #snap great analysis on #CNBC
输出：not sad

输入：You would think booking a holiday for 2 you'd be sat next to each other on the bloody plane #fuming 😡@ThomsonHolidays
输出：sad

输入：How come quiet well behaved cats and dogs have to ride on a plane in a tiny bag while screaming small humans roam free? #outrage #teampet
输出：not sad

输入：#depressed Today was bitter sweet watching all the kids go back to school made me really miss my babies. I'm so broke #backtoschool2017
输出：sad

输入：@o_pebbles Not of this one sadly! 😪
输出：sad

输入：My best friends driving for the first time with me in the car #terrifying
输出：sad

输入：@LoveMyFFAJacket FaceTime - we can still annoy you 😂
输出：not sad

输入：Migraine hangover has to be the worst thing ever 😣 #burst
输出：sad

输入：You can have a certain #arrogance, and I think that's fine, but what you should never lose is the #respect for the others.
输出：not sad

输入：Alright Alex and I have party boy neighbors who blast music 
输出：not sad

输入：Shooting more than ever, making more mistakes than ever but I jumped in the pool of sharks a long time ago.  #relentless *#resilient
输出：sad

输入：aleesha—kitchen sink, twenty one pilots (!!)"
输出：not sad

输入：@PuddlesPityP once again you have made me a very happy woman! Thx P and Casey. Sigh. #worththewait #cry #beyootifulll
输出：not sad

输入：@vivaonline Oh schade 🙁
输出：sad

输入：Saw my first Larsen trap today with stressed magpie. I NEVER EVER want to see that again  #angry #distressed #wildlife
输出：sad

输入：Theme of week: Ask the Lord for strength & perspective to persevere in #integrity and effort, despite being #disheartened & disappointed.
输出：sad

输入：@AllyiahsFace You're page is full of make up. It's a valid question. But you probably prefer to be smashed and dashed
输出：sad

输入：@hollloman @nessa_babbby @ajvannozzi97 As half a set of twins I resent that!
输出：not sad

输入：@CNN @NewDay If #trump #whitehouse aren't held accountable for their actions,what precedent is being set for future presidencies. #nightmare
输出：sad

输入：That moment when you look back and realise you've been a #selfish #horrible #judgemental person. #FeelingAshamed
输出：sad

输入：Listen ... this golden brown is giving me life ... but why the hell did my feet have to get so dark 😓
输出：sad

输入：Got woken up by a road sweeper I was trying to sleep 
输出：sad

输入：@silverstein 13th time seeing you guys today and you cancel the meet and greet because of the storm. We're all soaked.. 😡 #dissapointed
输出：sad

输入：@JohnMayer No DSM shows. #sadness
输出：sad

输入：@FabianisWailea No. No i did not. 
输出：not sad

输入：@Argos_Online customer service is dreadful, phone bill is huge and get passed from person 2 person and keep taking money off my card #idiots
输出：sad

输入：If u #smile too much😀\nu get frown lines,\nif u #cry too much😣 \nu get eye wrinkles,\n\n🔂 laugh&cry\n\n ...it's #life 💗\n\nHave a great day\n\n🍯🐝's
输出：not sad

输入：Woke up feeling fresh with a clear mind. That's never happened before.\n#morning #sober
输出：not sad

输入：By the way...in case you didnt know...joshuas goin out tonight...to take the trash out and then play blues at ever us 6.75
输出：not sad

输入：Julia and I are finally going to be able to meet PTX #crying
输出：sad

输入：@F1 Why announcing so late, it will be hard to make it from Manchester and organising a day off. #sad
输出：sad

输入：How shit and depressing is this weather wish I can travel the world for a living
输出：sad

输入：@Uber very disappointing that support has not responded to my email!! #bad #uber #service
输出：sad

输入：I think I'll eternally be irritated by our LIT teacher 😂
输出：not sad

输入：@JoshuaRozenberg Oh dear! #tantrums
输出：not sad

输入：@virtualalien there are more #frightening things in life\n\n#BeyondTheSphereOfReasonableDoubt
输出：sad

输入：#Worry, #doubt, #fear and #despair are the enemies which slowly bring us down to the ground and turn us to dust before we die.
输出：sad

输入：@hdfcergogic yr customer care exec r unable to pull information on l&t insurance #horrible #service #beware #renewal
输出：sad

输入：Freakin a...I deleted half the episode of BIP on accident  #bachelorinparadise
输出：not sad

输入：@mysarazaman Ya it makes me be more selfish and disheartened
输出：sad

输入：Quite possibly the worst anthems I've ever heard in pro sports by both singers. I wish I was deaf. #ASG2017 #mlb #MLBAllStarGame 
输出：sad

输入：My dog wouldn't stop barking so now I'm up at eight am with a raging headache
输出：sad

输入：@rosieblossoms_ What a miserable piece of shit 😡
输出：sad

输入：My boyfriend is my whole world 😭
输出：not sad

输入：smiling but we're close to tears
输出：sad

输入：Paul Ehrlich said that humanity is a threat to all life. Such great news to start the day. If I could blush I would.
输出：not sad

输入：@WDA_Punisher @SgtDangerCow @Battlefield I think so too. \nCasual audience was particularly unhappy with server admins and server rules.
输出：sad

输入：The way I'm always on twitter at work is a little alarming 🤦🏾‍♀️
输出：not sad

输入：@UKSportsZone In other words, I don't like the result of the poll so I'm packing up my polls & taking them home while I pout. 😂😂😻 #bbn
输出：not sad

输入：depression sucks😔
输出：sad

输入：A programmer’s wisdom is understanding the difference between getting program to run and having a runnable program.  #puppy
输出：not sad

输入：@GemmaAnneStyles @Fabicelolly we don't have these in america (i think) i'm #upset and #hurt
输出：sad

输入：Someone's nicked my lunch out the fridge at work!! Roast dinner as well! #fuming
输出：sad

输入：To #Wyoming: you have a beautiful state but your road signs, or lack thereof, are terrible. #lost
输出：sad

输入：Man I feel like crap today 😰
输出：sad

输入：If God had a plan he would've made already #discouraged
输出：sad

输入：Chelsea and united must be furious
输出：not sad

输入：@kylegriffin1 It's disgusting. #sick 
输出：sad

输入：@FFigureFBust How so? I've been thinking of getting it done and now you've given me a frighten.
输出：sad

输入：O, the melancholy Catacombs quickly wandered about the Rue Morgue, Madman!
输出：not sad

输入：Been at work for not even 4 hours and I've thrown boiling tea everywhere, smashed a mug, smashed a milk jug and sliced my finger open😐
输出：sad

输入：Threaten to leave your girl shaking in a wet spot ....
输出：not sad

输入：@ThomasEWoods I would like to hear a podcast of you going off refuting her entire article. Extra indignation please.
输出：not sad

输入：@chrisyour Don't be sad, ultimately it's apparently not supported completely but mostly working :-)
输出：not sad

输入：Didn't know the @ChickfilA cow day thing ended at 7:30 so showed up 30 min late looking like a cow with no sandwich #sadness 😅😅😅
输出：sad

输入：I woke up still not hardly believing what all happened last night 😰
输出：sad

输入：@BorisJohnson unhappy about the cost of leaving the EU ? If only you had published your letter outlining the reasons to stay instead. Idiot.
输出：sad

输入：@del_krushnic I knw you have a temper paaa lol just chill dear don't be pissed waii or I will else worry u saaa but I'm sorry 😫
输出：sad

输入：I'm back on Twitter! #madden #madden18 #maddenmobile #ea #nfl #espn #packers
输出：not sad

输入：Please ruin this party, @NSWRL. #origin #blues
输出：sad

输入：When did it all started? All these depression & anxiety shits?All these suicide thoughts?All of these bad thoughts thts hugging me at night?
输出：sad

输入：If you build up resentment in silence are you really doing anyone any favors
输出：not sad

输入：#Hopkinsville #Ky is #total #eclipse #capital of #world #August 2017 #dancing n #moon #dark #eclipse2017 #EclipseAcrossAmerica #Edgar #Cayce
输出：not sad

输入：I love seeing @yeahlizzy because it reminds me I'm not the only one miserable at work 😅
输出：sad

输入：Hmm...looks like no one @Tampax is available to respond. Its like waiting for a #doomsday reply. That #burning question with no #answer
输出：not sad

输入：@morninggloria I came for the 'damn boy' jokes but stayed because you are great at your actual job 👍🏾
输出：not sad

输入：Coulda sworn it was Interview With A Vampire. Hmmm......Mandela Effect anyone? \n#interviewwithavampire #annerice #books #horror #ilovevamps
输出：not sad

输入：@rachelkennedy84 @callummay @Ned_Donovan @JeyyLowe @jimwaterson @dats Can it do the bell ringing and incense at consecration?
输出：not sad

输入：@JeffBezos @amazon Who can I talk to about being terminated with no answer and not being paid for hours I worked? ERC have no answers 
输出：sad

输入：and after i got home in such a horrible mood my mom pissed me off the moment i stepped my feet in the house so i really almost go off on her
输出：sad

输入：@TerraJole is a bully.  plain and simple. 
输出：sad

输入：@LondonEconomic Sometimes our judiciary just leaves you breathless and speechless.
输出：sad

输入：#pause You gotta luv March Madness. Mad?? More like #upset
输出：sad

输入：Massive night tonight with the decider! Who you got? @QLDmaroons or @NSWRL ??? Can the #blues get it done?! #origin #StateOfOrigin
输出：not sad

输入：@RGUpdate Have you tried English hospital food ?? #yak #gross #horrible
输出：sad

输入：Morning Swindon!\nIs there any chance you can cheer yourself up a bit?!! #bleak 😝
输出：sad

输入：Don't be discouraged.
输出：not sad

输入：@shahidafridi37 at his best.... Great to watch u go big sir. #tremendous hits \nDts boom boom Afridi 😍
输出：not sad

输入：we mourn the death of our hopes today #james
输出：sad

输入：That eclipse sucked #dissapointed
输出：sad

输入："We just have to hold on a bit longer, then we can sink this monster into the ocean." #foodparty
输出：not sad

输入：Daniel killing her ex doeee🍫🍫🍫😛 #insecure
输出：sad

输入：Here is the message Cnbc is sending they Don't care that Melissa Lee has a horrible smile or a that Joe Kernan needs braces
输出：not sad

输入：sleep is and will always be one of the best remedies for a tired and weary soul
输出：sad

输入：Do #you #said, people never cross the 10! #serious
输出：not sad

输入：@taportugal I'm lost in translation with @taportugal #sad #tired #upset #noservice
输出：sad

输入：@narendramodi Really it was very sad and shame!!!
输出：sad

输入：Caroline FFS shut this whinging ‘Last Word Tom’ up!! He is unlikeable & a smartass! His biases R boring! @carolinemarcus @SkyNewsAust 
输出：not sad

输入：@patrickcafila September? Really? 😢
输出：sad

输入：I can drive you crazy without each other, chase each other but it was supposed to go again.\n I see you, my sadness, Be my
输出：sad

输入：@KatRamsland 'call to action by TV host John Oliver, who urged viewers to leave comments expressing their displeasure at the FCC's policies.
输出：sad

输入：Should I change my layout too? This one looks pretty depressed 😂
输出：sad

输入：Rin might ever appeared gloomy but to be a melodramatic person was not her thing.\n\nBut honestly, she missed her old friend. The special one.
输出：sad

输入：A joyless faith is not one for which Jesus died. #thegospel #joy #Jesus #happiness
输出：not sad

输入：Hello twitter! My next few tweets are going to be a bit gushy.... Look away if you're feeling bilious!!!
输出：not sad

输入：Sometimes you have to let the tears drop and realize tough times are temporary. Pick yourself up, stop wishing and start doing #toughday #😢
输出：sad

输入：@exceptions Although I have a nice vulva, I choose not to intimidate other women with it.
输出：not sad

输入：When you put vienesse whirls in the same tub as the horrid cherry Bakewell 😩😭😷 #grim
输出：sad

输入：@jimwilkz Because then the doom and gloom brigade would have nothing to moan about. We both know what a tragedy that would be!
输出：sad

输入：ENTRYLOG: CSEternity- With time on my hands for half an hour, I feel slightly melancholy, for some reason….\n  .
输出：not sad

输入：No, I'm not 'depressed because of the weather,' I'm depressed because I have #depression #sicknotweak
输出：sad

输入：Come on blues #StateOfOrigin #Origin  #NSWBlues #nsw
输出：not sad

输入：Beware of little expenses. A small leak will sink a great ship. -Benjamin Franklin
输出：not sad

输入：@Teenique yessss waiting for an epi is for the birds. it sucks. im waiting for walking dead new season😩 
输出：sad

输入：There are parts of you that wants the sadness. Find them out, ask them why
输出：sad

输入：Going to be a student nurse for 4 weeks eeeeppp #newexperience #scared #excitedmuch
输出：not sad

输入：feeling like a grim reaper all day hehehe\n9 days pa 🎩✉️
输出：not sad

输入：@ellagallagher Thing is tho my pout was actually serious
输出：not sad

输入：@Geordiegirl1967 @TKnicegirl @castielsmish they need time to grieve.
输出：sad

输入：My visit to hospital for care triggered #trauma from accident 20+yrs ago and image of my dead brother in it. Feeling symptoms of #depression
输出：sad

输入：@Aajijie Dont it'll hurts 😦 i prefer hugs hehe xoxo
输出：not sad

输入：Sometimes when I feel a bit depressed, I go back and watch the @Leighgriff09 free kicks against England to make me happy
输出：not sad

输入：@EdwardTHardy @realDonaldTrump You really need better source info and avoid the fake news, it has clouded even the obvious
输出：not sad

输入：@HughShows Ha, sadly not: just the undying respect of your peers, I'm afraid...
输出：sad

输入：@BBCNews @BBCBreaking I don't think it's very funny you guys bullying a man for struggling to put on a poncho #bully #bullies #bbc
输出：sad

输入：Lugubrious face, crestfallen eyes, forlorn heart and an agitated soul seeking serenity.
输出：sad

输入：Would love to stop crying sometime today, on my tenth cry 🙄 deep sadness on top of food poisoning do not mix. #miserable
输出：sad

输入：@responficient11 there would likely have been signs. But let her grieve. It's not your yen yen yen, after all.
输出：not sad

输入：@real_age I am so stressed today I have time so if they drop it, it would be so nice. 😢
输出：sad

输入：@pitchblacksteed Nana's death in the Royle Family 😢
输出：sad

输入：#upset #emotional\nMissing loved ones, wishing they were with me and this nightmare was over 😔😥
输出：sad

输入：'If I allow my worry to consume me - all that I worry about will be given the opportunity to manifest.' ~ #Eleesha ღ #quote #worry #trust
输出：sad

输入：@LadyScully I didn't. We all went out and got pissed down the local instead. 😄
输出：not sad

输入：I actually hate Vision. Coming over from @emiswebsupport is like moving from Man City to QPR. A league below, past it and clouded in chaos.
输出：sad

输入：@sundarpichai @Google Since when did someone break #chromecast for ALL MacBooks? PLS -We #despair at days wasted trying 2 make it work again
输出：sad

输入：@bierandcrumpets Never knew it was such an issue! Is there nowhere slightly dull but dependable - a kind of M&S?
输出：not sad

输入：@ReporterLaurenB @Examiner Agree with @Examiner or you are of a lower intelligence would be your message there then? #dreadful
输出：sad

输入：I thought the eclipse would make it not so damn hot outside today #disappointment
输出：sad

输入：i keep watching all the videos i took from two door cinema club and im sad they were so good i miss them
输出：sad

输入：If you let your insecurities get the best of you,  you will hurt someone unwillingly. I've learned that!  #tired #blah
输出：sad

输入：@IRP1916 do you have the first peppa pig vid with come out ye black and tans handy? obviously lost online with the old page :(
输出：not sad

输入：you take horrible dick pics @LeafyIsHere
输出：not sad

输入：@prisonerben1 Try explaining to Joe Public these charities do not simply give offenders 'treats' #despair
输出：sad

输入：Majka didn't start  :(
输出：sad

输入：@BdairAhmed @acmilan @VMontella :( hope to see niang start, then
输出：not sad

输入：if jk realize what taehyung did can he do that to taehyung as a revenge? i would love to
输出：not sad

输入：Last night was the first night I slept by myself in 3 weeks and it was awful.
输出：sad

输入：Preload on @CallofDuty WW2 begins and Ive entered preorder code but still have'nt received the beta start email!! #CallOfDuty #upset #gaming
输出：not sad

输入：Do not linger too long near the howff or you risk the displeasure of a chuhaister with pubes more underwhelming than those of an aurochs.
输出：not sad

输入：@ShaneMalwa @MajorPoonia @devyanidilli I believe ur mekka n madina also dikling which u kiss. A dark dirty one
输出：not sad

输入：Anytime I start getting sad about the fact that I might die alone I just look at how miserable 90% of my married friends are.
输出：sad
"""


# 6 个提示词 - Examples v13 终极版 (解决误报/漏报问题)
PROMPTS = [
    # Prompt 1: v13 精准版
    f"""判断推文作者是否表达负面情绪。

【宽泛悲伤定义】
悲伤、失望、委屈、沮丧、心碎、痛苦、愤怒、烦躁、厌恶、不满、难过

【立即排除 - 满足任一→not sad】
1. 纯积极：happy/love/good/awesome/lucky/congrats/achieved/worththewait
2. 成就场景 +😭: lucky/achieved/completed/exam done/favorite +😭
3. 礼貌用语："sorry" + welcome/thanks/greeting (无其他负面情绪)
4. 纯中性：无情绪表达的客观信息
5. 纯幽默：lol/haha/lmao/dumbest/goofy 且无真实负面情绪
6. 纯祝福：welcome/thanks/congrats
7. 宗教/引用：#written/#God/#mourn in #Zion/#wrath/#avenge
8. 讽刺语气：:) / "I guess" / "Oh dear" / 夸张表达
9. 纯标签：#pout/#lost/#worry/#tantrums/#tantrums 无真实负面情绪
10. 仅疲劳：tired 但无其他负面情绪词

【特别注意 - 以下情况→not sad】
- "pissy" + 无其他愤怒词 → not sad (轻微不满)
- "sorry" + welcome/greeting → not sad (礼貌用语)
- "unfortunately" + 无具体负面情绪 → not sad (客观)
- 表情 😩 + horny/sext/nudes 等 → not sad (非悲伤)

【确认 sad - 有以下任一→sad】
- 悲伤词：sad/crying/hurt/pain/depressed/lonely/unhappy/sorry(非礼貌)
- 愤怒词：angry/rage/furious/fuming/hate/horrible/dreadful/awful/horrid/rude
- 烦躁词：frustrated/annoyed/irritated/pissed(有愤怒) /infuriated
- 失望词：disappointed/let down/bad/worst/dismayed
- 痛苦词：hurting/pain/suffering/heartache
- 挫败词：broken/ruined/stuck/can't do + 负面情绪
- 强烈词：horrific/dreadful/awful/horrid
- 抱怨表达："needs more"/"doesn't"/"can't handle" + 负面情绪
- 第一人称 + 负面情绪："I'm"/"I feel"/"I was" + 情绪词
- 情绪标签："#sad"/"#angry"/"#disappointed"/"#frustrated"/"#unhappy"/"#rage"

【特殊规则】
- "unhappy 😂" → sad (有真实负面情绪 unhappy)
- "dad won't let me...😂" → not sad (幽默表达，无真实悲伤)
- "#Rage #disappointment...Lol" → sad (真实情绪标签 + 愤怒/失望词)
- "#sad" (有标签) → sad
- "hurting myself" → sad (真实伤害)

【FEW SHOT】
{FEW_SHOT}

只输出：sad 或 not sad

输入：{{text}}""",
    
    # Prompt 2: Few-shot v13
    f"""判断推文作者是否表达负面情绪。只输出 sad 或 not sad。

【宽泛悲伤定义】
悲伤、失望、委屈、沮丧、心碎、痛苦、愤怒、烦躁、厌恶、不满、难过

【立即排除】
1. 纯积极：happy/love/good/awesome/lucky/congrats/achieved
2. 成就场景 +😭: lucky/achieved/completed/favorite +😭
3. 礼貌用语："sorry" + welcome/thanks/greeting
4. 纯中性：无情绪表达的客观信息
5. 纯幽默：lol/haha/lmao/dumbest 且无真实负面情绪
6. 纯祝福：welcome/thanks/congrats
7. 宗教引用：#written/#God/#mourn in #Zion/#wrath
8. 讽刺语气：:) / "I guess" / "Oh dear"
9. 纯标签：#pout/#lost/#worry 无真实负面情绪
10. 仅疲劳：tired 无其他负面情绪

【特别注意→not sad】
- "pissy" + 无愤怒词 → not sad (轻微不满)
- "sorry" + welcome → not sad (礼貌)
- 😩 + horny/sext → not sad (非悲伤)

【示例学习】
示例 1: "U so lucky ahu 😭" → not sad (成就场景)
示例 2: "Past my test exam .. CDL achieved 😥" → not sad (成就)
示例 3: "Hello...Welcome...i'm deeply sorry for the late greeting" → not sad (礼貌 sorry)
示例 4: "Then why'd they wait until now to start getting pissy?" → not sad (pissy 轻微)
示例 5: "#dmme #kikme #horny...😩 horny" → not sad (非悲伤)
示例 6: "'#Dearly...#wrath: for it is #written'" → not sad (宗教)
示例 7: "#pout #heylookatthedistraction" → not sad (纯标签)
示例 8: "I'm kind of tired of everything #summer" → not sad (仅疲劳)
示例 9: "Grass growing simulator is offended" → sad (愤怒)
示例 10: "Your opinions on sports is dreadful" → sad (厌恶)
示例 11: "United Airline needs more Kiosks" → sad (抱怨)
示例 12: "#Rage and #disappointment man...." → sad (愤怒/失望标签)
示例 13: "unhappy and unfulfilled 😂" → sad (真实情绪 unhappy)
示例 14: "Had frustration dream...furious" → sad (愤怒)
示例 15: "I want to digital art so bad, but my dad won't let me use my iPad till exams are over 😂" → not sad (幽默，无真实悲伤)
示例 16: "I'm kind of tired of everything #summer #heat" → not sad (仅疲劳)
示例 17: "#sad" (有 sad 标签) → sad
示例 18: "Went to bed a 1:30...I'm dying... 😧" → sad (痛苦)

【FEW SHOT】
{FEW_SHOT}


判断流程:
礼貌/幽默/成就/宗教/标签/疲劳排除 → 负面情绪识别 → 确认 sad

只输出：sad 或 not sad

输入：{{text}}""",
    
    # Prompt 3: 严格版 v13
    f"""判断：作者是否表达负面情绪？

【立即排除（满足任一→not sad）】
1. 纯积极：happy/love/good/awesome/lucky/congrats/achieved → not sad
2. 成就场景 +😭: lucky/achieved/completed/favorite +😭 → not sad
3. 礼貌用语："sorry" + welcome/thanks/greeting → not sad
4. 纯中性：无情绪表达的客观信息 → not sad
5. 纯幽默：lol/haha/lmao/dumbest 且无真实负面情绪 → not sad
6. 纯祝福：welcome/thanks/congrats → not sad
7. 宗教引用：#written/#God/#mourn in #Zion/#wrath/#avenge → not sad
8. 讽刺语气：:) / "I guess" / "Oh dear" / 夸张表达 → not sad
9. 纯标签：#pout/#lost/#worry/#tantrums 无真实负面情绪 → not sad
10. 仅疲劳：tired 无其他负面情绪 → not sad

【特别注意→not sad】
- "pissy" + 无愤怒词 → not sad (轻微不满)
- "sorry" + welcome/greeting → not sad (礼貌用语)
- 😩 + horny/sext/nudes → not sad (非悲伤)

【确认 sad - 有以下任一→sad】
- 悲伤词：sad/crying/hurt/pain/depressed/lonely/unhappy/sorry(非礼貌)
- 愤怒词：angry/rage/furious/fuming/hate/horrible/dreadful/awful/horrid/rude
- 烦躁词：frustrated/annoyed/irritated/pissed(有愤怒)/infuriated
- 失望词：disappointed/let down/bad/worst/dismayed
- 痛苦词：hurting/pain/suffering/heartache
- 挫败词：broken/ruined/stuck/can't do + 负面情绪
- 强烈词：horrific/dreadful/awful/horrid
- 抱怨表达：needs more/can't handle/doesn't + 负面情绪
- 第一人称 + 负面情绪：I'm/I feel/I was + 情绪词
- 情绪标签：#sad/#angry/#disappointed/#frustrated/#unhappy/#rage

【FEW SHOT】
{FEW_SHOT}

【特殊规则】
- "unhappy 😂" → sad (真实情绪)
- "dad won't let me...😂" → not sad (幽默)
- "#Rage #disappointment...Lol" → sad (愤怒/失望标签)
- "#sad" (有标签) → sad
- "hurting myself" → sad (真实伤害)

只输出：sad 或 not sad

输入：{{text}}""",
    
    # Prompt 4: 分步版 v13
    f"""请分步骤判断作者是否表达负面情绪。

【步骤 1: 检查是否纯积极】
- 有 happy/love/good/awesome/lucky/congrats/achieved 且无负面情绪？→ not sad

【步骤 2: 检查是否成就场景 +😭】
- 有 lucky/achieved/completed/favorite +😭？→ not sad

【步骤 3: 检查是否礼貌用语】
- 有 "sorry" + welcome/thanks/greeting 且无其他负面情绪？→ not sad

【步骤 4: 检查是否纯中性】
- 无情绪表达的客观信息？→ not sad

【步骤 5: 检查是否纯幽默】
- 有 lol/haha/lmao/dumbest/goofy 且无真实负面情绪？→ not sad

【步骤 6: 检查是否纯祝福】
- 有 welcome/thanks/congrats？→ not sad

【步骤 7: 检查是否宗教引用】
- 有 #written/#God/#mourn in #Zion/#wrath/#avenge？→ not sad

【步骤 8: 检查是否讽刺语气】
- 有 :) / "I guess" / "Oh dear" / 夸张表达？→ not sad

【步骤 9: 检查是否纯标签】
- 有 #pout/#lost/#worry/#tantrums 无真实负面情绪？→ not sad

【步骤 10: 检查是否仅疲劳】
- 有 tired 但无其他负面情绪？→ not sad

【步骤 11: 特别注意排除】
- "pissy" + 无愤怒词？→ not sad
- 😩 + horny/sext/nudes？→ not sad

【步骤 12: 识别负面情绪词】
- 悲伤词：sad/crying/hurt/pain/depressed/lonely/unhappy/sorry(非礼貌)
- 愤怒词：angry/rage/furious/fuming/hate/horrible/dreadful/awful/horrid/rude
- 烦躁词：frustrated/annoyed/irritated/pissed(有愤怒)/infuriated
- 失望词：disappointed/let down/bad/worst/dismayed
- 痛苦词：hurting/pain/suffering/heartache
- 挫败词：broken/ruined/stuck/can't do

【步骤 13: 识别抱怨句式】
- needs more/can't handle/doesn't + 负面情绪 → 识别为负面情绪

【步骤 14: 识别第一人称表达】
- I'm/I feel/I was + 负面情绪词 → 识别为负面情绪

【步骤 15: 识别情绪标签】
- #sad/#angry/#disappointed/#frustrated/#unhappy/#rage → 识别为负面情绪

【步骤 16: 特殊判断】
- "unhappy 😂" → sad
- "dad won't let me...😂" → not sad
- "#Rage #disappointment...Lol" → sad
- "#sad" (有标签) → sad
- "hurting myself" → sad

【FEW SHOT】
{FEW_SHOT}

【最终判断】
步骤 1-11 任一通过 → not sad
否则 步骤 12-16 任一识别为负面情绪 → sad

只输出：sad 或 not sad

输入：{{text}}""",
    
    # Prompt 5: 权重版 v13
    """判断推文作者是否表达负面情绪。

【情绪权重评分】
+3 分：强烈负面情绪 (rage/furious/fuming/horrible/dreadful/horrific)
+2 分：明显负面情绪 (angry/disappointed/frustrated/hurt/pain/depressed/lonely/unhappy)
+1 分：轻微负面情绪 (bad/worst/sad/unfulfilled/annoyed)
+1 分：抱怨表达 (needs more/can't handle/doesn't)
+1 分：第一人称 + 负面情绪 (I'm/I feel + 情绪词)
+1 分：情绪标签 (#sad/#angry/#disappointed/#rage)
+2 分：混合情绪 +😂: "unhappy 😂" → sad

-3 分：成就场景 +😭: lucky/achieved +😭
-3 分：礼貌用语："sorry" + welcome/thanks/greeting
-2 分：纯积极 (happy/love/good/awesome/lucky)
-2 分：纯中性 (事实陈述)
-2 分：纯幽默 (lol/haha/lmao)
-2 分：纯祝福 (welcome/thanks/congrats)
-3 分：宗教引用 (#written/#God/#mourn in #Zion)
-2 分：讽刺语气 (:)/I guess/Oh dear)
-2 分：纯标签 (#pout/#lost/#worry)
-1 分：疲劳 (tired 仅身体)
-1 分："pissy" + 无愤怒词
-2 分：😩 + horny/sext/nudes

【判定规则】
总分 ≥ 0 分 → sad
总分 < 0 分 → not sad

只输出：sad 或 not sad

输入：{text}""",
    
    # Prompt 6: 综合版 v13
    f"""判断推文作者是否表达负面情绪。只输出 sad 或 not sad。

【宽泛悲伤定义】
悲伤、失望、委屈、沮丧、心碎、痛苦、愤怒、烦躁、厌恶、不满、难过

【立即排除 - 满足任一→not sad】
- 纯积极：happy/love/good/awesome/lucky/congrats/achieved
- 成就场景 +😭: lucky/achieved/completed/favorite +😭
- 礼貌用语："sorry" + welcome/thanks/greeting
- 纯中性：无情绪表达的客观信息
- 纯幽默：lol/haha/lmao 且无真实负面情绪
- 纯祝福：welcome/thanks/congrats
- 宗教引用：#written/#God/#mourn in #Zion/#wrath
- 讽刺语气：:) / I guess / Oh dear / 夸张表达
- 纯标签：#pout/#lost/#worry/#tantrums 无真实负面情绪
- 仅疲劳：tired 无其他负面情绪

【特别注意→not sad】
- "pissy" + 无愤怒词 → not sad (轻微不满)
- 😩 + horny/sext/nudes → not sad (非悲伤)

【确认 sad - 有以下任一→sad】
- 悲伤词：sad/crying/hurt/pain/depressed/lonely/unhappy/sorry(非礼貌)
- 愤怒词：angry/rage/furious/fuming/hate/horrible/dreadful/awful/horrid/rude
- 烦躁词：frustrated/annoyed/irritated/pissed(有愤怒)/infuriated
- 失望词：disappointed/let down/bad/worst/dismayed
- 痛苦词：hurting/pain/suffering/heartache
- 挫败词：broken/ruined/stuck/can't do + 负面情绪
- 强烈词：horrific/dreadful/awful/horrid
- 抱怨句式：needs more/can't handle/doesn't + 负面情绪
- 第一人称 + 负面情绪：I'm/I feel/I was + 情绪词
- 情绪标签：#sad/#angry/#disappointed/#frustrated/#unhappy/#rage

【特殊规则】
- "unhappy 😂" → sad
- "dad won't let me...😂" → not sad
- "#Rage #disappointment...Lol" → sad
- "#sad" (有标签) → sad
- "hurting myself" → sad

判断流程:
礼貌/幽默/成就/宗教/标签/疲劳/特别注意排除 → 负面情绪识别 → 确认 sad

【FEW SHOT】
{FEW_SHOT}

只输出：sad 或 not sad

输入：{{text}}"""
]

SYSTEM_PROMPT = "你是一位情感分类专家，请根据提示词判断文本是否表达负面情绪。只输出 sad 或 not sad。"

def get_prediction(text, prompt,temperature):
    content = prompt.format(text=text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ]
    
    try:
        res = client.chat.completions.create(model="/Qwen3-4B/Qwen/Qwen3-4B", messages=messages, temperature=temperature)
        output = res.choices[0].message.content.lower().strip()
        if 'sad' in output and 'not sad' not in output:
            return 'sad'
        return 'not sad'
    except:
        return None

# 修改 1：增加 task_id 参数，并返回字典而不是元组
def process_single_sample(sample, system_prompt, task_id):
    """处理单条数据的逻辑"""
    sample_id = sample.get("id", "") # 获取单条数据的内部 ID（如果有的话）
    input_text = clean_text(sample["input"]) # 对输入文本进行清洗，去除 @ 用户名等
    votes = {'sad': 0, 'not sad': 0}
    predictions = []
    n_models = len(PROMPTS)  # 使用所有提示词进行预测
    for i in range(min(n_models, len(PROMPTS))):
        pred = get_prediction(input_text, PROMPTS[i], 0.0)
        if pred:
            votes[pred] += 1
            predictions.append(pred)
    
    if votes['sad'] > votes['not sad']:
        prediction = 'sad'
    else:
        prediction = 'not sad'
    
    # 将需要保存的所有信息打包成一个字典返回
    return {
        "task_id": task_id,
        "sample_id": sample_id,
        "input": input_text,
        "model_output": prediction,
    }

if __name__ == "__main__":
    file_path = r"D:\WorkSpace\python\flagOS赛题三\LongContext-ICL-Annotation\data\openseek-5_semeval_2018_task1_tweet_sadness_detection.json"
    out_path = r"D:\WorkSpace\python\flagOS赛题三\LongContext-ICL-Annotation\rgs_q5\experiment/openseek-5-v1.jsonl"
    
    task5_id, examples, test_samples = task5_data_loader(file_path)
    system_prompt = "你是情感分析专家，能识别文字背后的真实情绪。特别注意讽刺、幽默、表情符号的语境含义。" # 系统提示词经实验无效
    eval_data = test_samples
    total_cnt = len(eval_data)

    # 准备一个 partial 函数，固定住 system_prompt 和 task_id，方便 map 调用
    process_func = partial(process_single_sample, system_prompt=system_prompt, task_id=task5_id)
    
    max_workers = 200 
    results_list = []
    correct_cnt = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用 executor.map 替代 as_completed，以保证返回顺序与 eval_data 完全一致
        # executor.map 会自动并发，但按输入顺序 yield 结果
        for result in tqdm(executor.map(process_func, eval_data), total=total_cnt, desc="Evaluating"):
            results_list.append(result)

# 提取 sample_id 和 model_output
    output_data = [
        {
            "test_sample_id": item["sample_id"],
            "prediction": item["model_output"]
        }
        for item in results_list
    ]

    # 确保输出路径的文件夹存在
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in output_data:
            # 将单个字典转换为 JSON 字符串，然后手动添加换行符
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + '\n')

    print(f"评测结果已成功保存为 JSONL 格式：{out_path}")
        