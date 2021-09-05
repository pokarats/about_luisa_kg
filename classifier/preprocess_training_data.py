from collections import defaultdict

question_dict = defaultdict(list)

with open("responses.txt", encoding="UTF-8") as file1:
    for line in file1.readlines():
        ques_ans = line.split(sep=" <answer> ")
        try:
            question_dict[ques_ans[0]].append((ques_ans[1].strip(), 0))
        except IndexError:
            print(ques_ans)

print("part 2")
with open("responses_ann.csv", encoding="UTF-8", errors="ignore") as file2:
    for line in file2.readlines():
        try:
            ann, ques_ans = line.split(sep=";")
            ques_ans = ques_ans.split(sep=" <answer> ")
            question_dict[ques_ans[0]] = [(ans[0], ann) for ans in question_dict[ques_ans[0]]]
        except ValueError:
            print("x")

with open("responses_ann_new.txt", "w", encoding="UTF-8") as ans_file:
    for ques in question_dict.keys():
        for ans in question_dict[ques]:
            ans_file.write(f"{ques} {ans[0]};{ans[1]}\n")
