# 此前需要先确定有feed
import xgboost
import lightgbm as lgb
from sklearn import metrics
from sklearn import model_selection
import config
import pandas as pd
import numpy as np
import sys

# 定义基础模型函数


class model_demo():
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_dir = config.model_dir+model_name+'/'
        config.mkdirector(self.model_dir)
        self.result_dir = config.result_dir+model_name+'.csv'

    def read_csvs(self):
        self.columns = pd.read_csv(config.train_feed).columns
        self.train_feed = pd.read_csv(config.train_feed).astype(float).values
        self.train_target = pd.read_csv(config.train_target).astype(int).values.flatten()
        self.test_feed = pd.read_csv(config.test_feed).astype(float).values
        self.test_target = pd.read_csv(config.test_target).astype(int).values.flatten()
        self.submit_feed = pd.read_csv(config.submit_feed).astype(float).values

    def train(self):  # 获取模型
        raise NotImplementedError

    def test(self):  # 测试模型
        raise NotImplementedError

    def get_result(self):  # 获取预测结果
        raise NotImplementedError

    def save(self):  # 保存结果
        result = pd.read_csv(config.submit_data)[['id']]
        result['target'] = pd.Series(self.result)
        result.to_csv(self.result_dir, index=False)

    def run_all(self):  # 完整的从模型训练，测试以及结果提交
        self.read_csvs()
        self.train()
        self.test()
        self.result = self.get_result()
        self.save()


class xgb_model(model_demo):
    def __init__(self, grid_paras, model_name):
        super().__init__(model_name)
        self.grid_paras = grid_paras
        self.run_all()

    def train(self):
        model = xgboost.XGBClassifier()
        param_search = self.grid_paras
        searcher = model_selection.GridSearchCV(estimator=model, param_grid=param_search, cv=4, n_jobs=-1)
        searcher.fit(self.train_feed, self.train_target)
        print(searcher.best_params_, searcher.best_score_)
        self.model = searcher.best_estimator_

    def test(self):
        train_predict = self.model.predict_proba(self.train_feed)[:, 1]
        train_auc = metrics.roc_auc_score(self.train_target, train_predict)
        test_predict = self.model.predict_proba(self.test_feed)[:, 1]
        test_auc = metrics.roc_auc_score(self.test_target, test_predict)
        print(train_auc, ' ', test_auc)

    def get_result(self):
        submit_predict = self.model.predict_proba(self.submit_feed)[:, 1]
        return submit_predict


class lgb_model(model_demo):
    def __init__(self, params, model_name):
        super().__init__(model_name)
        self.params = params
        self.run_all()

    def train(self):
        train_data = lgb.Dataset(self.train_feed, label=self.train_target.flatten(), feature_name=list(self.columns))
        # test_data = lgb.Dataset(self.test_feed, label=self.test_target, feature_name=list(self.columns))
        bst = lgb.cv(self.params, train_data, num_boost_round=1000, nfold=3, early_stopping_rounds=30)
        estimators = lgb.train(self.params, train_data, num_boost_round=len(bst['auc-mean']))
        self.model = estimators

    def test(self):
        train_predict = self.model.predict(self.train_feed)
        train_auc = metrics.roc_auc_score(self.train_target, train_predict)
        test_predict = self.model.predict(self.test_feed)
        test_predict = self.ajust(test_predict)
        test_auc = metrics.roc_auc_score(self.test_target, test_predict)
        print(train_auc, ' ', test_auc)

    def get_result(self):
        submit_predict = self.model.predict(self.submit_feed)
        return submit_predict

    def ajust(self, predict):
        result = []
        for num in list(predict):
            if num < 0:
                result.append(0)
            elif num > 0:
                result.append(1)
            else:
                result.append(num)
        return np.array(result)


if __name__ == '__main__':
    xgb_origin_grid_paras = {
        'max_depth': list(range(4, 5, 1)),
        'colsample_bytree': [0.4, 0.5, 0.6],
        'min_child_weight': [4, 6, 8],
        'reg_alpha': [0.2, 1, 5],
        'reg_lambda': [5, 10, 20],
        'learning_rate': [0.08, 0.1, 0.12],
        'max_delta_step': [3, 5, 7]
    }
    xgb_origin = xgb_model(xgb_origin_grid_paras, 'xgb_origin')
    lgb_origin_param = {
        'max_depth': 6,
        'num_leaves': 64,
        'learning_rate': 0.03,
        'scale_pos_weight': 1,
        'num_threads': 40,
        'objective': 'binary',
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'min_sum_hessian_in_leaf': 100
    }
    # lgb_origin = lgb_model(lgb_origin_param, 'lgb_origin')
    pass
