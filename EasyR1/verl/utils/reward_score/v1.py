import re
import json


def extract_thinking_answer_regex(s):
    s = s.strip()
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, s, re.DOTALL)

    if match:
        ans = match.group(1).strip()
    else:
        return ""
    # print(s, '\n', ans)
    if len(ans.split()) > 10 and not re.search("[ABCD]", ans):
        return ""

    matches = re.search(r"[ABCD]", ans)
    if matches is None:
        return ""

    return matches[0]



def v1_compute_score(predict_str: str, ground_truth: list) -> float:
    score = 0.0
    try:
        content_answer = extract_thinking_answer_regex(predict_str)
    
        if content_answer.lower() in [gt.lower() for gt in ground_truth]:
            score = 1.0
        else:
            score = 0.0
        return score
    except Exception as e:
        print(e)
        pass  # Continue to next verification method if this fails

    return 0.0


# tt="<answer>D: 1. A smartphone interface showing the security system panel listing connected devices; 2. A wide shot of an outdoor security camera installed with greenery; 3. A close-up of a white circular security camera next to decorative figurines; 4. A close-up of a black device with the 'vivint' branding and transition to the red-circled 'c|net' logo.(main_task pid=11307) </answer>"
# print(extract_thinking_answer_regex(tt))
