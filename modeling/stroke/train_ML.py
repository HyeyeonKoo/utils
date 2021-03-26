#-*-coding:utf-8-*-

"""
트리 기반 ML 모델을 학습시켰을 때, XGBoost가 점수는 가장 높음(과적합도 제일 덜 됨)
그러나, 테스트 샘플에서 1의 비율이 워낙 작으므로 1의 Precision, Recall, FScore 점수가 굉장히 낮게 나옴
=> 전처리에서 SMOTE를 학습 데이터에만 사용했으므로 그랬을 수도 있음
"""


import csv
import numpy as np
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_fscore_support
import joblib


"""
의사결정트리 : 매 분기마다 변수 영역을 두 개로 구분하는 모델
    불순도 또는 불확실성을 최소화하는 방향으로 학습 진행
"""
def train_decesion_tree(train_data, train_label, test_data, test_label):
    start = datetime.now()
    
    model = DecisionTreeClassifier(
        random_state=111,
        criterion="entropy",    # gini : 0.86 / entropy : 0.88
    )
    model.fit(train_data, train_label)

    print("[Decesion Tree]")
    evaluating(model, train_data, train_label, test_data, test_label)
    
    end = datetime.now()
    print("Time : " + str(end-start))
    print("\n")

    return model


"""
랜덤포레스트 : 여러 개의 의사결정트리를 생성하고, 각 트리가 분류한 결과에서 가장 득표 수가 높은 결과를 최종 분류 결과로 선택
    Bagging : 중복을 허용한 부분집합을 추출해 각 다른 의사결정트리를 학습
"""
def train_random_forest(train_data, train_label, test_data, test_label):
    start = datetime.now()
    
    model = RandomForestClassifier(
        random_state=111,
        criterion="entropy", 
        n_estimators=10
    )
    model.fit(train_data, train_label)

    print("[Random Forest]")
    evaluating(model, train_data, train_label, test_data, test_label)
    
    end = datetime.now()
    print("Time : " + str(end-start))
    print("\n")    

    return model


"""
그래디언트 부스팅 : 하나의 의사결정트리에서 값을 예측한 뒤, 그 결과에 따라 데이터에 가중치를 부여하고, 그 다음 트리를 만드는 방식
    Boosting :  처음 모델이 예측을 하면 그 예측 결과에 따라 데이터에 가중치가 부여되고 부여된 가중치가 다음 모델에 영향을 줌,
                잘못 분류된 데이터에 집중하여 새로운 분류 규칙을 만드는 단계를 반복함
"""
def train_gradient_boosting(train_data, train_label, test_data, test_label):
    start = datetime.now()
    
    model = GradientBoostingClassifier(
        random_state=111,
        max_depth=10,
    )
    model.fit(train_data, train_label)

    print("[Gradient Boosting]")
    evaluating(model, train_data, train_label, test_data, test_label)
    
    end = datetime.now()
    print("Time : " + str(end-start))
    print("\n")

    return model


"""
XGBoost : 기본 그래디언트 부스팅 목적식에 제약식을 추가해 과적합을 방지
"""
def train_XGBoost(train_data, train_label, test_data, test_label):
    start = datetime.now()
    
    model = XGBClassifier(
        random_state=111
    )
    model.fit(np.array(train_data), np.array(train_label))

    print("[XGBoost]")
    evaluating(model, train_data, train_label, test_data, test_label)
    
    end = datetime.now()
    print("Time : " + str(end-start))
    print("\n") 

    return model    


"""
LightGBM : XGBoost보다 뿌리를 조금 더 깊게 내릴 수 있음
        (XGBoost : Level단위 학습 / LightGBM : Leaf단위 학습)
        training loss를 더 줄일 수 있지만, 과적합이 더 잘 됨
        XGBoost에 비해 속도가 빠르고 GPU를 지원하는 장점이 있음
"""
def train_LGBM(train_data, train_label, test_data, test_label):
    start = datetime.now()
    
    model = LGBMClassifier(
        random_state=111
    )
    model.fit(np.array(train_data), np.array(train_label))

    print("[Light GBM]")
    evaluating(model, train_data, train_label, test_data, test_label)
    
    end = datetime.now()
    print("Time : " + str(end-start))
    print("\n")

    return model    


def evaluating(model, train_data, train_label, test_data, test_label):
    print("Train Score : {:.2f}".format(model.score(train_data, train_label)))
    print("Test Score : {:.2f}".format(model.score(test_data, test_label)))

    data = precision_recall_fscore_support(
        test_label, model.predict(test_data)
    )

    print("\t\t0\t1")
    print("Count\t\t" + str(round(data[3][0], 2)) + "\t" + str(round(data[3][1], 2)))
    print("Precesion\t" + str(round(data[0][0], 2)) + "\t" + str(round(data[0][1], 2)))
    print("Recall\t\t" + str(round(data[1][0], 2)) + "\t" + str(round(data[1][1], 2)))
    print("Fscore\t\t" + str(round(data[2][0], 2)) + "\t" + str(round(data[2][1], 2)))  


def get_csv(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for line in csv_reader:
            data.append(list(map(float, line)))

    return data    


def get_data(train_data_path, train_label_path, test_data_path, test_label_path):
    train_data = get_csv(train_data_path)
    train_label = get_csv(train_label_path)

    test_data = get_csv(test_data_path)
    test_label = get_csv(test_label_path)

    return (
        np.array(train_data), np.array(train_label), 
        np.array(test_data), np.array(test_label)
    )


if __name__=="__main__":
    train_data, train_label, test_data, test_label = get_data(
        "data/train_data.csv",
        "data/train_label.csv",
        "data/test_data.csv",
        "data/test_label.csv"
    )

    decesion_tree_model = train_decesion_tree(
        train_data, train_label, test_data, test_label
    )

    joblib.dump(decesion_tree_model, "model/decesion_tree.pkl")

    random_forest_model = train_random_forest(
        train_data, train_label, test_data, test_label
    )
    joblib.dump(random_forest_model, "model/random_forest.pkl")

    gradient_boosting_model = train_gradient_boosting(
        train_data, train_label, test_data, test_label
    )
    joblib.dump(gradient_boosting_model, "model/gradient_boosting.pkl")

    XGBoost_model = train_XGBoost(
        train_data, train_label, test_data, test_label
    )
    joblib.dump(XGBoost_model, "model/XGBoost.pkl")

    LGBM_model = train_LGBM(
        train_data, train_label, test_data, test_label
    )
    joblib.dump(LGBM_model, "model/LGBM.pkl")


