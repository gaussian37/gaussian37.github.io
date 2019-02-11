---
layout: post
title: 이미지 One Class Classification with Deep features
date: 2019-02-11 00:00:00
img: dl/concept/one-class/one-class.png
categories: [dl-concept] 
tags: [deep learning, one class, one class svm, resnet, Isolation Forest, Gaussian Mixture] # add tag
---

<br>

## One Class SVM과 Isolation Forest 모델을 이용한 구현

+ 먼저 Imagedata set을 이용하여 학습한 ResNet-50 feature를 계산합니다.

```python
from keras.applications.resnet50 import ResNet50
def extract_resnet(X):  
    # X : images numpy array
    resnet_model = ResNet50(input_shape=(image_h, image_w, 3), 
							weights='imagenet', include_top=False)  
	# Since top layer is the fc layer used for predictions
    features_array = resnet_model.predict(X)
    return features_array
```

<br>

+ 모든 이미지 데이터 셋에서 ResNet feature를 계산할 수 있습니다.
+ 그 다음 계산을 하여 얻은 feature에 `standard scaler`를 적용합니다.
+ PCA를 적용합니다. 이 때 `n_components=512`로 해보겠습니다.
+ PCA 적용 한 후 남은 feature를 `One Class SVM` 또는 `Isolation Forest`에 전달합니다.

+ 아래 코드를 보면 X_train과 X_test는 train과 test 이미지의 ResNet feature 입니다.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm

# Apply standard scaler to output from resnet50
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)

# Take PCA to reduce feature space dimensionality
pca = PCA(n_components=512, whiten=True)
pca = pca.fit(X_train)
print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# Train classifier and obtain predictions for OC-SVM
oc_svm_clf = svm.OneClassSVM(gamma=0.001, kernel='rbf', nu=0.08)  # Obtained using grid search
if_clf = IsolationForest(contamination=0.08, max_features=1.0, max_samples=1.0, n_estimators=40)  # Obtained using grid search

oc_svm_clf.fit(X_train)
if_clf.fit(X_train)

oc_svm_preds = oc_svm_clf.predict(X_test)
if_preds = if_clf.predict(X_test)

# Further compute accuracy, precision and recall for the two predictions sets obtained
```

<br>

+ One-Class SVM 또는 Isolation Forest 모두 -1 또는 1 값을 반환합니다.
	+ 1 인 경우는 One Class에 해당합니다. Positive
	+ -1인 경우는 One Class에 해당하지 않습니다. Negative

<br><br>

## Gaussian Mixture와 Isotonoic Regression을 사용한 One Class Classification

+ 음식 사진들이 주어졌을 때 음식 사진들은 다양한 음식의 cluster에 속할 수 있습니다.
+ 어떤 음식 사진은 동시에 여러개의 cluster에 속할 수도 있습니다.
+ 결과적으로 이런 경우에 `Gaussian mixture`를 Positive class data points(ResNet Feature)에 학습시킬 수 있습니다.
	+ `Gaussian mixture`는 확률 모델 종류 중의 하나로 전체 모수 내에서 normal distribution을 가지는 부분 모집단을 표현하는 모델입니다.
	+ `Gaussian Mixture Model`은 데이터에 의해 학습이 되면 **새로운 데이터가 입력 되었을 때 이 데이터가 현재 학습된 데이터의 분포에서 부터 생성**될 수 있는지의 확률을 만들어 냅니다.
