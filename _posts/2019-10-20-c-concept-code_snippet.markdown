---
layout: post
title: C 언어 코드 snippet
date: 2019-10-20 00:00:00
img: c/c.png
categories: [c-concept] 
tags: [C] # add tag
---

<br>

## **목차**

<br>

- ### 매크로 모음
- ### 변수명 출력
- ### 숫자 → 문자열
- ### 문자열 → 숫자
- ### 정수 → 바이너리 문자열
- ### 텍스트 입/출력 
- ### 파일 존재하는 지 확인
- ### 기본 문자열 관련 함수
- ### 특정 문자를 기준으로 문자열 split
- ### 부분 문자열 (substr) 저장
- ### 부분 문자열 (substr) 출력
- ### 현재 시간 출력

<br>

## **머신러닝 / 딥러닝 관련 모음**

<br>

- ### softmax 함수
- ### entropy 함수

<br>

## **매크로 모음**

<br>

- 타입에 상관없이 사용할 수 있는 함수 형태의 매크로 또는 자주 사용할 수 있는 상수들을 모았습니다.

<br>

```cpp
#define PI 3.141592
#define TRUE 1
#define FALSE 0
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define ABS(x) ((x)<0 ? -(x) : (x))
#define INF 999999999
```

<br>

## **변수명 출력**

<br>

```cpp
#include <stdio.h>

 #define GET_VARIABLE_NAME(variable, holder) sprintf(holder, "%s", #variable)

int main() {
    char variable_name[100]; // 출력할 변수명을 저장할 문자열
    GET_VARIABLE_NAME(print_this_variable, variable_name);
    puts(variable_name); // print_this_variable이 출력됩니다.
    return 0;
}
```

<br>

## **숫자 → 문자열**

<br>

- 숫자를 문자로 변경할 때에는 `sprintf` 함수를 사용합니다.

<br>

```c
#include <stdio.h>
#include <stdlib.h>

int main(){

    int a=123;
    int b=0;
    char buf[10];
    
    sprintf(buf, "%d", a);
    printf("%s\n",buf); 
```

<br>

## **문자열 → 숫자**

<br>

- 문자열을 숫자로 바꿀 때에는 아래 함수를 사용하여 바꿀 수 있습니다.

```c

int StringToInteger(char *str) {
    int ret = 0;
    while (*str) {
        ret = ret * 10 + (int)(*str - '0');
        str++;
    }
    return ret;
}

```

<br>

## **정수 → 바이너리 문자열**

<br>

- 아래 코드는 정수를 입력 받아서 바이너리 형태로 출력하는 코드 입니다. 그 정수의 이진수 값을 알고 싶을 때 사용하시면 됩니다.

<br>

```c
void PrintIntToBinary(int n){

    char str[100];
    int i = 0;

    printf("%d : ", n);
    do{
        if (n % 2 == 0){
            str[i] = '0';
        }
        else{
            str[i] = '1';
        }
        n /= 2;
        i++;
    }while(n > 0);

    while(i > 0){
        printf("%c", str[--i]);
    }
    printf("\n");
}
```

<br>

## **텍스트 입/출력**

<br>

- FILE 포인터와 fprintf를 이용하여 텍스트 파일에 출력하는 방법입니다.
- 아래 코드와 같이 사용 시 fprintf 함수의 파일 포인터에 해당하는 파일에는 fprintf를 통해 출력한 값만 적혀집니다.
- 물론 printf를 통하여 출력한 출력은 콘솔창에 기존 그대로 찍히게 됩니다.

<br>

```c
#include <stdio.h>

int main(){
    // "a" 적용 시 파일 끝에서 부터 계속 이어서 쓴다.
    // "w" 적용 시 파을을 덮어쓰기 한다.
    FILE* fp = fopen("output.txt", "w"); 
    fprintf(fp, "%d %lf\n", 1, 2.0);
    
    return 0;
}
```

<br>

- 이번에는 반대로 FILE 포인터와 fscanf를 이용하여 텍스트 파일을 읽어서 입력하는 방법입니다.
- 기본적인 사용법은 scanf와 유사하며 파라미터로 File Pointer를 넘겨주는 것에 차이가 있습니다.

<br>

```c
#include <stdio.h>

int main(){
    // "a" 적용 시 파일 끝에서 부터 계속 이어서 쓴다.
    // "w" 적용 시 파을을 덮어쓰기 한다.
    int a;
    double b;
    FILE* fp = fopen("input.txt", "r"); 
    fscanf(fp, "%d %lf\n", &a, &b);
    
    return 0;
}
```

<br>

- `fscanf` 함수가 더 이상 읽을 것이 없으면 -1을 반환합니다.
- 아래 코드와 같이 `fscanf`가 -1을 반환할 때 까지 파일을 읽으면 됩니다.

<br>

```c
#include <stdio.h>
#include <stdlib.h>

int main() {

	FILE* fp = fopen("input.txt", "r");
	int a;
	int ret = 1;
	while (1) {
		ret = fscanf(fp, "%d", &a);		
		if (ret < 0) {
			break;
		}
		printf("%d\n", a);
	}
}
```

<br>

## **파일 존재하는 지 확인**

<br>

- 다음은 파일 경로를 입력하였을 때, 그 파일이 존재하는 지 확인하는 코드 입니다.

<br>

```c
// 입력 받은 경로에 파일이 존재하는 지 확인한다.
int FileExists(const char *file_name){
    FILE *file;
    int ret;
    if ((file = fopen(file_name, "r"))){
        fclose(file);
        ret = 1;
    }
    else{
        ret = 0;
    }

    return ret;
}
```

<br>

## **기본 문자열 관련 함수**

<br>

- `strlen` : 문자열 길이 반환
    - 예) `strlen(str)`
- `strcmp` : 문자열 비교
    - 예) `strcmp(str1, str2)`
- `strncmp` : 문자열 n개 비교
    - 예) `strcmp(str1, str2, 8)`
- `strcpy` : 문자열을 복사하는 함수 인자로 주소값을 받습니다.
    - 예) `strcpy(dest, src)`. 여기서 dest는 복사를 할 곳의 시작 `주소`이고 src는 복사할 문자열의 시작 `주소`입니다.
    - 따라서 문자열의 시작 주소를 적절하게 응용하여 사용하면 좋습니다.
- `strncpy`: 문자열 n개 복사
    - 예) `strncpy(dest, src, 8)`
- `strcat` : 새로운 문자열을 기존 문자열 끝에 붙입니다. 따라서 기존 문자열의 NULL을 찾고 그 위치에서 부터 새로운 문자열을 붙입니다.
    - 예) `strcat(dest, src)`
- `strncat` : 새로운 문자열을 기존 문자열 끝에 n개 붙입니다.
    - 예) `strncat(dest, src, 8)`





<br>

## **특정 문자를 기준으로 문자열 split**

<br>

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// str배열을 delim의 문자들로 split한 후 tokenize 배열에 시작 인덱스들을 저장하고 저장된 인덱스의 갯수를 반환한다.
// 사용 방법 : StringTokenizer(str, delim, tokenize, sizeof(tokenize)/sizeof(int));
int StringTokenizer(const char* str, const char* delim, int* tokenize, int max_tokenize_index){
    
    int token_count = 0;
    int str_len = strlen(str);    
    
    // initialize string and tokenize
    char *s;    
    s = (char*)malloc(sizeof(char)*(str_len + 1));
    strcpy(s, str);
    memset(tokenize, -1, (int)sizeof(tokenize));

    char *start_point = s;
    char *ptr = strtok(s, delim);
    
    while (ptr != NULL){

        if(token_count >= max_tokenize_index){
            printf("The number of token exceeds maximum index of tokenize array.\n");
            exit(1);
        }
        tokenize[token_count++] = (int)(ptr - start_point); 		
		ptr = strtok(NULL, delim);
    }

    tokenize[token_count] = str_len;

    free(s);
    return token_count;
}
```

<br>

## **부분 문자열 (substr) 저장**

<br>

- 단순히 부분문자열을 구하려면 `strncpy` 함수를 사용 하면 됩니다.
    - `strncpy(dest, str + 시작인덱스, 길이)`
- 범위를 이용하여 부분 문자열을 구하려면 다음 함수를 사용하면 됩니다. `SubstringToDest` 함수는 파라미터로 받은 `dest` 배열에 부분 문자열을 저장하는 형태이고 `GetSubstring`은 함수 내에서 동적 할당을 하여 반환값으로 새로운 부분 문자열을 반환하는 방법입니다. 범위는 \[begin, end) 입니다.

<br>

```c
// str 문자열에서 [begin, end) 범위의 부분 문자열을 반환한다.
// malloc을 사용하였으므로 free로 해제해주어야 한다.
char* GetSubstring(const char *str, int begin, int end){	
    char *ret;
    int len = end - begin;
    ret = (char*)malloc(sizeof(char) * (len + 1));	
    strncpy(ret, (str + begin), len);	
	return ret;
}

// str 문자열에서 [begin, end) 범위의 부분 문자열을 dest에 저장한다.
void SubstringToDest(const char* str, char* dest, int begin, int end){
    strncpy(dest, str + begin, end - begin);
}
```

<br>

## **부분 문자열 (substr) 출력**

<br>

- 부분 문자열을 출력하는 함수 입니다.

<br>

```c
void PrintSubstring(const char* str, int begin, int end){
    for(int i = begin; i < end; ++i){
        printf("%c", str[i]);
    }
    printf("\n");
}
```

<br>

## **현재 시간 출력**

<br>

- C에서 현재 시간을 출력할 때, `time.h` 헤더의 `time_t` 와 `localtime 함수`를 사용할 수 있습니다.
- 아래 코드를 사용하면 year, month, day, hour, minute, second 를 각각 출력할 수 있습니다.

<br>

```c
time_t t = time(NULL);
struct tm tm = *localtime(&t);
printf("%d-%d-%d %d:%d:%d\n", tm.tm_year+1900, tm.tm_mon+1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
```

<br>

- `tm` 구조체에 대하여 알아보면 다음과 같습니다. 타입은 모두 정수형 입니다.

<br>

```c
struct tm {
   int tm_sec;         /* second,  range 0 to 59            */
   int tm_min;         /* minute, range 0 to 59             */
   int tm_hour;        /* hour, range 0 to 23            */
   int tm_mday;        /* day, range 1 to 31             */
   int tm_mon;         /* month, range 0 to 11             */
   int tm_year;        /* year, from 1900년                */
   int tm_wday;        /* day of the week, range sunday(0) to saturday(6) */
   int tm_yday;        /* elapsed day of year, range 0 to 365  */
   int tm_isdst;       /* summer time                        */
};
```

<br>

## **머신러닝 / 딥러닝 관련 모음**

<br>

## **softmax 함수**

<br>

- 아래 코드는 (h, w, c) 크기의 이미지가 있을 때, 채널 c 방향으로 softmax를 적용하는 함수 입니다.

<br>

```c

#define MAX(a,b) (((a)>(b))?(a):(b))

void set_softmax(float probablities[], int num_classes) {
	float max_value = -999;
	float denominator = 0;
	for (int i = 0; i < num_classes; ++i) {
		max_value = MAX(max_value, probablities[i]);
	}
	for (int i = 0; i < num_classes; ++i) {
		probablities[i] -= max_value;
		probablities[i] = expf(probablities[i]);
		denominator += probablities[i];
	}
	for (int i = 0; i < num_classes; ++i) {
		probablities[i] /= denominator;
	}
}

```

<br>


