from wonderwords import RandomWord

r = RandomWord()

with open('noun_65536.txt', 'w') as f:
    for _ in range(65536):
        noun = r.word(include_parts_of_speech=["nouns"])
        f.write(noun + '\n')