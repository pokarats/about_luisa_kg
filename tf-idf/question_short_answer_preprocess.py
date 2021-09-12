
with open('question_short_answer.txt', 'w', encoding="ISO-8859-1") as writer:
    with open("qa_extended_pair.txt", 'r', encoding="ISO-8859-1") as f:
        for line in f:
            line = line.strip()
            line = line.split("<start extention>")
            question_short_answer = line[0]
            writer.write(question_short_answer + "\n")