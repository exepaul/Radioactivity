import unicodedata
import sys
import uuid






def remove_pun(text_):
    data=dict.fromkeys([i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')])

    return text_.translate(data)



sentences=[]
labels=[]


with open('movies_genres.csv','r') as f:
    for line_np,line in enumerate(f):
        uni_id=uuid.uuid4()
        if line_np==10:
            break
        else:
            if line_np==0:
                pass

            else:
                data=line.split('\t')
                sentences.append([str(uni_id),remove_pun(data[1])])

                sentences.append([str(uni_id),[int(i.strip()) for i in data[2:]]])

print(sentences)
print(labels)





