# 此前需要先确定有feed
import xgboost
from sklearn import metrics
from sklearn import model_selection
import config
import pandas as pd
import numpy as np

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
        self.train_target = pd.read_csv(config.train_target).astype(int).values
        self.test_feed = pd.read_csv(config.test_feed).astype(float).values
        self.test_target = pd.read_csv(config.test_target).astype(int).values
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
        model = xgboost.XGBRegressor(objective='reg:squarederror')
        param_search = self.grid_paras
        searcher = model_selection.GridSearchCV(estimator=model, param_grid=param_search, cv=2, n_jobs=-1)
        searcher.fit(self.train_feed, self.train_target)
        self.model = searcher.best_estimator_

    def test(self):
        test_predict = self.model.predict(self.test_feed)
        test_auc = metrics.roc_auc_score(self.test_target, test_predict)
        print(test_auc)

    def get_result(self):
        submit_predict = self.model.predict(self.submit_feed)
        return submit_predict


if __name__ == '__main__':
    xgb_origin_grid_paras = {
        'max_depth': list(range(3, 4, 1))
    }
    xgb_origin = xgb_model(xgb_origin_grid_paras, 'xgb_origin')
    pass
