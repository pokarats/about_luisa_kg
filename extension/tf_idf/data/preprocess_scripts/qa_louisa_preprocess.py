import re
from collections import defaultdict

QA_dict = defaultdict(list)

with open("3000QuestionsAndAnswers.txt", 'r', encoding="ISO-8859-1") as f:
    for line in f:
        line = line.strip()
        search = re.match("^\d+.(.+)?$", line)
        if search:
            question = search.group(1).strip()
        else:
            answer = line.replace("-", "").strip()
            QA_dict[question].append(answer)

# print(QA_dict['when were you born?'])

with open('responses.txt', 'w', encoding="ISO-8859-1") as writer:
    for key, value in QA_dict.items():
        writer.write(key + '<answer>' + '<answer>'.join(value) + "\n")
