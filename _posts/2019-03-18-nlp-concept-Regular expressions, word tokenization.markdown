---
layout: post
title: Regular expressions & word tokenization
date: 2019-03-18 00:10:00
img: nlp/concept/nlp.jpg
categories: [nlp-concept] 
tags: [nlp, Regular expressions, word tokenization] # add tag
---

+ 정규식표현에 대하여 알아보도록 하겠습니다.
+ 실습 예제로 정규식 표현은 다음과 같이 사용합니다.

```python
import re
re.findall(PATTERN, string)
```

+ 위의 명령어에서 `re`는 `regular expression` 패키지를 뜻합니다.
+ `re.findall(PATTERN, string)`의 첫번째 파라미터인 `pattern`에는 정규식문법이 들어가고 정규식문법을 통하여 원하는 문자를 검출해냅니다.

```python
my_string = "Let's write RegEx!"
>> re.findall(r"\w+", my_string)
['Let', 's', 'write', 'RegEx']
```

+ 위 코드에서 사용된 `r"\w+"`에서 `\w`는 문자를 뜻하고 `+`는 하나 또는 그 이상 연결된을 뜻합니다. 접두사인 `r`은 정규식표현임을 뜻합니다.
  + 따라서 `r"\w+"`는 하나 이상의 문자로 연결된 문자열을 뜻합니다.

### 정규식 표현 예제

```python
# Import the regex module
import re

# Write a pattern to match sentence endings: sentence_endings
sentence_endings = r"[.,?,!]"

# Split my_string on sentence endings and print the result
print(re.split(sentence_endings, my_string))

# Find all capitalized words in my_string and print the result
capitalized_words = r"[A-Z]\w+"
print(re.findall(capitalized_words, my_string))

# Split my_string on spaces and print the result
spaces = r"\s+"
print(re.split(spaces, my_string))

# Find all digits in my_string and print the result
digits = r"\d+"
print(re.findall(digits, my_string))
```

