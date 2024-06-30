import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns

# 读取训练集
train_X = np.load('./data/train/train_minist.npy')  # 数字矩阵
train_label = pd.read_csv('./data/train/train_label.csv')
train_number = train_label['number']  # 数字标签
train_size = train_label['size']  # 粗细标签
# 读取测试集
test_X = np.load('./data/test/test_minist.npy')
test_label = pd.read_csv('./data/test/test_label.csv')
test_number = test_label['number']
test_size = test_label['size']
# 查看数据集规模
print(f"训练集的尺度是：{train_X.shape}, 测试集的尺度是：{test_X.shape}")


# ----------------------------->第一题（必做）
# 1:使用Logistic回归拟合训练集的X数据和size标签,并对测试集进行预测
#
train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)

def problem1():
    # 初始化参数
    w = np.zeros(train_X.shape[1])
    b = 0
    alpha = 0.01  # 学习率
    epochs = 1000  # 迭代次数

    # Sigmoid 函数
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # 训练 Logistic 回归模型
    for epoch in tqdm(range(epochs)):
        linear_model = np.dot(train_X, w) + b
        y_pred = sigmoid(linear_model)

        # 计算梯度
        dw = (1 / train_X.shape[0]) * np.dot(train_X.T, (y_pred - train_size))
        db = (1 / train_X.shape[0]) * np.sum(y_pred - train_size)

        # 更新参数
        w -= alpha * dw
        b -= alpha * db

    # 预测测试集
    linear_model_test = np.dot(test_X, w) + b
    y_pred_test = sigmoid(linear_model_test)
    y_pred_class = np.where(y_pred_test > 0.5, 1, 0)

    # 计算 Accuracy, Precision, Recall
    TP = np.sum((y_pred_class == 1) & (test_size == 1))
    TN = np.sum((y_pred_class == 0) & (test_size == 0))
    FP = np.sum((y_pred_class == 1) & (test_size == 0))
    FN = np.sum((y_pred_class == 0) & (test_size == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # 计算 auROC
    roc_auc = roc_auc_score(test_size.to_numpy(), y_pred_test)

    # 输出结果
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1_score:.4f}')
    print(f'auROC: {roc_auc:.4f}')

    # 画出 ROC 曲线
    fpr, tpr, _ = roc_curve(test_size.to_numpy(), y_pred_test)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

#
# ---------------------------->第二题（必做）
# 2:使用Softmax回归拟合训练集的X数据和number标签,并对测试集进行预测
#


def problem2():
    # 对标签进行 one-hot 编码
    encoder = OneHotEncoder()
    y_train_onehot = encoder.fit_transform(train_number.to_numpy().reshape(-1, 1)).toarray()
    y_test_onehot = encoder.transform(test_number.to_numpy().reshape(-1, 1)).toarray()

    # 初始化参数
    n_classes = y_train_onehot.shape[1]
    n_features = train_X.shape[1]
    W = np.zeros((n_features, n_classes))
    b = np.zeros(n_classes)
    alpha = 0.01  # 学习率
    epochs = 100  # 迭代次数

    # Softmax 函数
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 防止溢出
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # 训练 Softmax 回归模型
    for epoch in tqdm(range(epochs)):
        linear_model = np.dot(train_X, W) + b
        y_pred = softmax(linear_model)

        # 计算梯度
        dw = (1 / train_X.shape[0]) * np.dot(train_X.T, (y_pred - y_train_onehot))
        db = (1 / train_X.shape[0]) * np.sum(y_pred - y_train_onehot, axis=0)

        # 更新参数
        W -= alpha * dw
        b -= alpha * db

    # 预测测试集
    linear_model_test = np.dot(test_X, W) + b
    y_pred_test = softmax(linear_model_test)
    y_pred_class = np.argmax(y_pred_test, axis=1)

    # 计算各项指标
    # accuracy = accuracy_score(test_number, y_pred_class)
    # macro_precision = precision_score(test_number, y_pred_class, average='macro')
    # macro_recall = recall_score(test_number, y_pred_class, average='macro')
    # macro_f1 = f1_score(test_number, y_pred_class, average='macro')

    # 转换为 DataFrame
    results = pd.DataFrame({'y_true': test_number, 'y_pred': y_pred_class})

    # 计算每个类别的 TP, TN, FP, FN
    def calculate_metrics(results, n_classes):
        metrics = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
        for i in range(n_classes):
            TP = len(results[(results['y_true'] == i) & (results['y_pred'] == i)])
            TN = len(results[(results['y_true'] != i) & (results['y_pred'] != i)])
            FP = len(results[(results['y_true'] != i) & (results['y_pred'] == i)])
            FN = len(results[(results['y_true'] == i) & (results['y_pred'] != i)])
            metrics['TP'].append(TP)
            metrics['TN'].append(TN)
            metrics['FP'].append(FP)
            metrics['FN'].append(FN)
        return metrics

    metrics = calculate_metrics(results, n_classes)

    # 计算 Accuracy, Macro Precision, Macro Recall, Macro F1-score
    accuracy = np.sum(metrics['TP']) / len(test_label)

    precision_per_class = [metrics['TP'][i] / (metrics['TP'][i] + metrics['FP'][i])
                           if (metrics['TP'][i] + metrics['FP'][i]) > 0 else 0 for i in range(n_classes)]
    recall_per_class = [metrics['TP'][i] / (metrics['TP'][i] + metrics['FN'][i])
                        if (metrics['TP'][i] + metrics['FN'][i]) > 0 else 0 for i in range(n_classes)]
    f1_per_class = [2 * (precision_per_class[i] * recall_per_class[i]) / (precision_per_class[i] + recall_per_class[i])
                    if (precision_per_class[i] + recall_per_class[i]) > 0 else 0 for i in range(n_classes)]

    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(test_number, y_pred_class)

    # 输出结果
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Macro Precision: {macro_precision:.4f}')
    print(f'Macro Recall: {macro_recall:.4f}')
    print(f'Macro F1-score: {macro_f1:.4f}')

    # 画出混淆矩阵
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # 计算 auROC
    # 对于多分类问题，需要对每个类计算 ROC AUC，并取平均值
    roc_auc = roc_auc_score(y_test_onehot, y_pred_test, average='macro')

    # 输出 auROC
    print(f'auROC: {roc_auc:.4f}')


if __name__ == '__main__':
    problem1()
    problem2()
