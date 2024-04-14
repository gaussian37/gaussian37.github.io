---
layout: post
title: C++ 관련 유용한 코드 snippets
date: 2019-08-07 00:00:00
img: cpp/cpp.png
categories: [c-cpp] 
tags: [cpp, c++, 객체 지향, oop, object oriented] # add tag
---

<br>

- 이번 글에서는 C++ 관련 코드 사용 시 유용하게 사용하였던 코드들을 모아서 정리해 보도록 하겠습니다.

<br>

## **목차**

<br>

- ### nlohmann-json을 이용한 Json 파일을 읽기

<br>

## **nlohmann-json을 이용한 Json 파일을 읽기**

<br>

- 관련 깃 페이지 : https://github.com/nlohmann/json
- 참조 : https://snowdeer.github.io/c++/2022/01/11/cpp-nlohmann-json-example/
- 리눅스 설치 방법 : `sudo apt install nlohmann-json`

<br>

- `C++`에서는 간단한 `json` 파일을 읽는 것도 파이썬과 다르게 다소 까다롭습니다. 이번 글에서는 C++을 사용할 때 `nlohmann-json`을 이용하여 `json` 파일을 읽어서 간단히 다루는 예제를 살펴보도록 하겠습니다.
- `nlohmann-json`을 이용하여 `json` 파일을 읽으면 `python`의 `json.load()`를 하여 데이터를 `dictionary`로 읽은 구조와 거의 유사하게 사용할 수 있습니다.
- 아래 `nlohmann-json`을 사용하는 예제에서는 

<br>

- dkfo

```python
# filename.json 

{
    "A" : {
        "A1" : [1, 2, 3],
        "A2" : [4, 5, 6]
    },
    "B" : [7, 8, 9],
    "C" : 10
}
```

<br>

```cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <nlohmann/json.hpp>

using namespace std;

nlohmann::json read_json_file(const string& file_name){
    nlohmann::json json_object;
    ifstream jsonFile(file_name);

    if (jsonFile.is_open()){
        stringstream buffer;
        buffer << jsonFile.rdbuf();

        jsonb_object = nlohmann::json::parse(buffer.str());
    }
    else{
        cout << "Unable to open JSON file." << endl;
    }
    return json_object;
}

int main(){
    const string json_path = "PATH/TO/THE/filename.json";
    nlohmann::json json_object = read_json_file(json_path);

    cout << json_object["A"] << endl;
    // {"A1" : [1,2,3],"A2":[4,5,6]}
    cout << json_object["A"]["A1"] << endl;
    // [1,2,3]
    cout << json_object["B"] << endl;
    // [7,8,9]
    cout << json_object["C"] << endl;
    // 10
}
```

<br>

- 위 예시와 같이 python의 `dictionary` 구조와 동일하게 사용할 수 있습니다.