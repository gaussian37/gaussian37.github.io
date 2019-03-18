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

+ `my_string`이 다음과 같은 문자열이라고 가정해 보겠습니다.
    + "Let's write RegEx!  Won't that be fun?  I sure think so.  Can you find 4 sentences?  Or perhaps, all 19 words?"    

```python
# Import the regex module
import re

# Write a pattern to match sentence endings: sentence_endings
sentence_endings = r"[.,?,!]"

# Split my_string on sentence endings and print the result
>> print(re.split(sentence_endings, my_string))

["Let's write RegEx", "  Won't that be fun", '  I sure think so', '  Can you find 4 sentences', '  Or perhaps', ' all 19 words', '']

# Find all capitalized words in my_string and print the result
capitalized_words = r"[A-Z]\w+"
>> print(re.findall(capitalized_words, my_string))

['Let', 'RegEx', 'Won', 'Can', 'Or']

# Split my_string on spaces and print the result
spaces = r"\s+"
>> print(re.split(spaces, my_string))

["Let's", 'write', 'RegEx!', "Won't", 'that', 'be', 'fun?', 'I', 'sure', 'think', 'so.', 'Can', 'you', 'find', '4', 'sentences?', 'Or', 'perhaps,', 'all', '19', 'words?']

# Find all digits in my_string and print the result
digits = r"\d+"
>> print(re.findall(digits, my_string))

['4', '19']

```

+ 정규식표현 관련 대표적인 함수
+ `re.findall(패턴, 문자열)` : 정규식 표현에 부합하는 모든 문자열을 찾습니다.
+ `re.split(패턴, 문자열)` : 정규식 표현에 부합하는 문자를 기준으로 문자열을 자릅니다.

+ 정규식을 사용할 때, 대표적으로 사용하는 패턴은 다음고 같습니다.
+ `"\w"` : 알파벳 + 숫자
+ `"\d"` : 숫자
+ `"\s"` : 공백문자
+ `"+"` : 하나 이상
+ `"[]"` : 패턴을 직접적으로 명시할 때

