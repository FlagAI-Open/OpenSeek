# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    if method == "strict":
        # Strict: extract content of the last \boxed{...} (align with evaluator's parser)
        if "boxed" not in solution_str:
            final_answer = None
        else:
            ans = solution_str.split("boxed")[-1]
            if len(ans) == 0:
                final_answer = None
            elif ans[0] == "{":
                stack = 1
                a = ""
                for c in ans[1:]:
                    if c == "{":
                        stack += 1
                        a += c
                    elif c == "}":
                        stack -= 1
                        if stack == 0:
                            break
                        a += c
                    else:
                        a += c
                final_answer = a.replace(",", "").replace("$", "")
            else:
                a = ans.split("$")[0].strip()
                final_answer = a.replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(solution_str, ground_truth, method="strict", format_score=0.2, score=1.0):
    """GSM8K reward with \\boxed{} strict parsing and numeric equivalence.

    Scoring policy:
    - Strict format (\\boxed{...}) and numerically correct -> score (default 1.0)
    - Else if strict format present (even if wrong) -> format_score (default 0.2)
    - Else -> 0.0
    """
    # Extract strict (format-aware) and flexible answers
    answer_strict = extract_solution(solution_str=solution_str, method="strict")
    has_strict_format = answer_strict is not None

    # Numeric equivalence helper
    def _to_float(x):
        try:
            return float(str(x).replace(",", "").replace("$", "").strip())
        except Exception:
            return None

    def _num_equal(a, b):
        av = _to_float(a)
        bv = _to_float(b)
        if av is not None and bv is not None:
            return av == bv
        return str(a).strip() == str(b).strip()

    # 1) Strict-correct → full score
    if answer_strict is not None and _num_equal(answer_strict, ground_truth):
        return float(score)

    # 2) Format-only bonus if strict format present
    if has_strict_format:
        return float(0.2)

    # 3) No reward
    return 0.0
