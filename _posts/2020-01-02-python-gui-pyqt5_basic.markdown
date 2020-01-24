---
layout: post
title: PyQT5 기본 문법들
date: 2019-12-09 00:00:00
img: python/gui/gui.png
categories: [python-gui] 
tags: [python, pyinstaller] # add tag
---

<br>

- 설치 : `pip install pyqt5`
- 물론 설치는 가상 환경을 새로 만들어서 설치하는 것을 권장드립니다.

<br>

- 간단한 예제를 통하여 아래 내용을 살펴보도록 하겠습니다.
- ### Window 생성
- ### Label Widget
- ### Buttons
- ### LineEdit Widget
    - ID, 비밀번호 입력
- ### CheckBox
- ### 폴더 디렉토리 설정

<br>
<center><img src="../assets/img/python/gui/pyqt5_basic/0.png" alt="Drawing" style="width: 300px;"/></center>
<br> 

```python
import sys
from PyQt5.QtWidgets import *

class Window(QWidget):
    def __init__(self):
        super().__init__()
        # 화면의 (x : 150, y: 50) 위치에서 (width : 300, height : 450) 크기의 윈도우가 생성됩니다.
        self.setGeometry(150, 50, 300, 450)
        # Window의 Title을 지정합니다.
        self.setWindowTitle("윈도우의 제목")
        self.UI()

    def UI(self):
        # QLabel은 GUI에 고정된 텍스트를 입력합니다.
        self.text = QLabel("아이디와 비밀번호를 입력하세요", self)
        # QWidget 에서 상속받은 객체들은 move 함수를 가지고 있고 
        # 이 함수는 각 객체의 위치 (x좌표, y좌표)를 나타냅니다.
        self.text.move(50, 50)       

        # QLineEdit는 사용자로 부터 문자열을 입력 받도록 합니다.
        self.nameTextBox = QLineEdit(self)
        # QLineEdit의 setPlaceholderText는 QLineEdit의 기본 값을 나타냅니다.
        self.nameTextBox.setPlaceholderText("ID를 입력하세요")
        self.nameTextBox.move(50, 100)

        self.pwdTextBox = QLineEdit(self)
        self.pwdTextBox.setPlaceholderText("비밀번호를 입력하세요")
        # QLineEdit의 setEchoMode는 입력 받은 문자열을 비밀번호 처리하듯 보여줍니다. 
        self.pwdTextBox.setEchoMode(QLineEdit.Password)
        self.pwdTextBox.move(50, 150)

        # QPushButton은 클릭할 수 있는 버튼을 만듭니다.
        self.enterButton = QPushButton("입력", self)
        self.enterButton.move(50, 200)
        # QPushButton.clicked.connect를 통하여 버튼을 클릭하였을 때 
        # 동작해야 할 함수를 지정할 수 있습니다.
        self.enterButton.clicked.connect(self.getValues)
        
        # Window를 띄웁니다.
        self.show()

    def getValues(self):
        name = self.nameTextBox.text()
        password = self.pwdTextBox.text()
        self.text.setText("ID : " + name + ", Password : " + password)

def main():
    App = QApplication(sys.argv)
    window = Window()
    sys.exit(App.exec_())

if __name__ == "__main__":
    main()
```

<br>

- 이번에는 이미지를 삽입하는 방법을 다루어 보겠습니다.

<br>
<center><img src="../assets/img/python/gui/pyqt5_basic/1.png" alt="Drawing" style="width: 800px;"/></center>
<br> 

```python
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap

class Window(QWidget):
    def __init__(self):
        super().__init__()
        # 화면의 (x : 150, y: 50) 위치에서 (width : 300, height : 450) 크기의 윈도우가 생성됩니다.
        self.setGeometry(150, 50, 1000, 450)
        # Window의 Title을 지정합니다.
        self.UI()

    def UI(self):
        # 이미지를 삽일 할 때에도 QLabel을 이용합니다.
        self.image = QLabel(self)
        # QLabel의 속성에 보면 setPixmap이 있는데 이 값을 통하여 이미지를 지정합니다.
        self.image.setPixmap(QPixmap("image.png"))
        self.image.move(150, 50)

        remove_button = QPushButton("remove", self)
        remove_button.move(380, 400)
        remove_button.clicked.connect(self.Remove)

        show_button = QPushButton("show", self)
        show_button.move(500, 400)
        show_button.clicked.connect(self.Show)

        # Window를 띄웁니다.
        self.show()

    def Remove(self):
        self.image.close()

    def Show(self):
        self.image.show()

def main():
    App = QApplication(sys.argv)
    window = Window()
    sys.exit(App.exec_())

if __name__ == "__main__":
    main()
```

<br>

- `checkbox`를 사용하려면 다음과 같이 `UI`에 입력하면 됩니다.

<br>

```python
self.checkbox = QCheckBox("This is checkbox", self)
```

<br>

- 그리고 이 체크박스가 체크되었는 지 아닌 지 확인 하려면 다음과 같이 확인하면 됩니다.
- 즉, 어떤 함수에서 다음과 같이 작성하면 체크박스 여부에 따라 프로그램을 작성해 나아갈 수 있습니다.

<br>

```python
if (self.checkbox.isChecked()):
    print("Checkbox is checked")
```

<br>

- `폴더 디렉토리`를 설정하는 방법에 대하여 알아보겠습니다.
- 기본적으로 다음 명령어를 입력하면 폴더를 선택할 Window가 생성됩니다. (윈도우 이름에 아래 문자열과 같이 입력됩니다. Select Directory)
- 폴더를 선택하면 그 폴더의 경로가 문자열로 저장됩니다.

<br>

```python
path_name = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
```

<br>

- 실질적으로 사용하려면 메뉴나 버튼을 눌렀을 때, 이벤트 식으로 위 명령어가 실행되도록 하면 됩니다.
