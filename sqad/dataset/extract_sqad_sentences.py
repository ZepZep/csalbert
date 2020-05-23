import re
import pandas as pd

savename = "sqad_split.csv"
PATH = "/nlp/projekty/sqad/sqad_v3/data"
max_index = 13473
#PATH = "dataset/extract_test"
#max_index = 5

fwm = re.compile("^[^\\s]*")


def get_text_from_vert(path):
    sentences = []
    sentence = []
    glue = False
    with open(path) as f:
        for line in f:
            if line[:2] == "<s":
                sentence = []
            elif line[:3] == "</s":
                sentences.append(" ".join(sentence))
                sentence = []
            elif line[:2] == "<g":
                if not sentence:
                    continue
                glue = True
                continue
            elif line[0] == "<":
                continue
            else:
                match = fwm.search(line)
                word = match.group(0)
                
                if not word:
                    continue
              
                if glue:
                    sentence[-1] += word
                else:
                    sentence.append(word) 
            glue = False
        if sentence:
            sentences.append(" ".join(sentence))
    
    return "\n".join(sentences)

    
def get_row(i):
    text = get_text_from_vert(f"{PATH}/{i:06d}/03text.vert")
    question = get_text_from_vert(f"{PATH}/{i:06d}/01question.vert")
    answer = get_text_from_vert(f"{PATH}/{i:06d}/09answer_extraction.vert")
    answer_sentence = get_text_from_vert(f"{PATH}/{i:06d}/06answer.selection.vert")

    #occurences = [m.span() for m in re.finditer(answer, text)]
    #sentence_occurences = [m.span() for m in re.finditer(answer_sentence, text)]
    
    return text, question, answer, answer_sentence




df = pd.DataFrame(columns=["text", "question", "answer", "answer_sentence"])
try:
    for i in range(1, max_index+1):
        if not i % 10:
            print(f"{i:06d}/{max_index:06d}\r", end="", flush=True)
        row = get_row(i)
        df.loc[i] = row
except:
    print()
    df.to_csv(savename)
    raise
print()
df.to_csv(savename)



