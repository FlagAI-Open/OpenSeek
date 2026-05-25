from collections import Counter

def find_repeat_substr_by_rolling_hash(s: str, min_len = 3, max_len = None, only_max_sub = True) -> list:
    total_len = len(s)
    if max_len == None or max_len <= min_len:
        max_len = int(total_len // 2)

    max_sub_len = 0
    indexs = {}
    seens = {}
    subs = []
    for i,c in enumerate(s):
        if seens.get(c) == None:
            seens[c] = [ i ]
        else:
            for j in seens[c]:
                max_for_len = min(i-j, total_len-i)
                if max_for_len >= min_len:
                    max_for_len = min(max_for_len, max_len)
                    if not only_max_sub or max_for_len > max_sub_len:
                        sub = None
                        for l in range(min_len-1, max_for_len):
                            sub = s[j: j+l+1]
                            if sub != s[i: i+l+1]:
                                break
                        if sub != None:
                            cur_sub_len = len(sub)
                            if not only_max_sub or cur_sub_len >= max_sub_len:
                                max_sub_len = cur_sub_len

                                subs.append(sub)

            seens[c].append(i)

    return subs


if __name__=="__main__":

    # s = "abcdabcd"
    # r = find_repeat_substr_by_rolling_hash(s)
    # print(r)  # 输出: ['abcd']

    # r = find_repeat_substr_by_rolling_hash('term_term_term_term_term_action_type')
    # print(r)  # 输出: ['term_', 'term_term_', 'term_term_t']

    s = "I can't believe I'm going to miss them. I'm so sad. I'm so tired. I'm so tired of this. I'm so tired of this life. I'm so tired of this pain. I'm so tired of this sadness."

    arr = find_repeat_substr_by_rolling_hash(s, 10, 100)
    content_counter = Counter(arr)
    print('content_counter', content_counter)
