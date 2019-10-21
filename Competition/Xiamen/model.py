# 此前需要先确定有feed
import xgboost
from sklearn import metrics
from sklearn import model_selection
from sklearn import ensemble
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
        self.train_feed = pd.read_csv(config.train_feed_b).astype(float).values
        self.train_target = pd.read_csv(config.train_target_b).astype(int).values.flatten()
        #self.train_target = np.random.randint(0, 2, size=self.train_target.shape)

        self.test_feed = pd.read_csv(config.test_feed).astype(float).values
        self.test_target = pd.read_csv(config.test_target).astype(int).values.flatten()
        #self.test_target = np.random.randint(0, 2, size=self.test_target.shape)

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
    def __init__(self, model_name):
        super().__init__(model_name)
        self.run_all()

    def train(self):
        model = xgboost.XGBRegressor()
        self.model = model.fit(self.train_feed, self.train_target)

        '''
        param_search = {'max_depth':[4,5,6]}
        searcher = model_selection.GridSearchCV(estimator=model, param_grid=param_search, cv=5, n_jobs=-1)
        searcher.fit(self.train_feed, self.train_target)
        print(searcher.best_params_, searcher.best_score_)
        self.model = searcher.best_estimator_
        '''
        '''
        eval_set = [(self.train_feed, self.train_target), (self.test_feed, self.test_target)]
        model = xgboost.XGBRegressor(objective='binary:logistic')
        self.model = model.fit(self.train_feed, self.train_target, eval_metric='auc', eval_set=eval_set, verbose=True, num_boost_round=1000, early_stopping_rounds=200)
        '''
        '''
        self.train_matrix = xgboost.DMatrix(self.train_feed, label=self.train_target, feature_names=self.columns)
        self.test_matrix = xgboost.DMatrix(self.test_feed, label=self.test_target, feature_names=self.columns)
        self.submit_matrix = xgboost.DMatrix(self.submit_feed, feature_names=self.columns)
        watch_list = [(self.train_matrix, 'train'), (self.test_matrix, 'test')]
        
        self.model = xgboost.train({'subsample': 1.0, 'l1': 10, 'l2': 50, 'max_delta_step': 20, 'objective': 'binary:logistic', 'max_depth': 3, 'scale_pos_weight': 20, 'eta': 0.15},
                                   self.train_matrix, feval=self.auc_feval, evals=watch_list, num_boost_round=2000, early_stopping_rounds=30)
        # self.model = xgboost.train({'objective': 'binary:logistic'}, self.train_matrix, feval=self.auc_feval, evals=watch_list, num_boost_round=2000, early_stopping_rounds=30)
        
        '''

    def test(self):
        train_predict = self.model.predict(self.train_feed)
        train_auc = metrics.roc_auc_score(self.train_target, train_predict)
        test_predict = self.model.predict(self.test_feed)
        test_auc = metrics.roc_auc_score(self.test_target, test_predict)

        print(train_auc, ' ', test_auc)

    def get_result(self):
        submit_predict = self.model.predict(self.submit_feed)
        return submit_predict

    def auc_feval(self, preds, xgbtrain):
        label = xgbtrain.get_label()
        score = -metrics.roc_auc_score(label, preds)  # 需要添加负数
        return 'myFeval', score

    def reg_objedt(self):
        pass


class rd_forest(model_demo):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.run_all()

    def train(self):
        model = ensemble.RandomForestRegressor()
        self.model = model.fit(self.train_feed, self.train_target)

    def test(self):
        train_predict = self.model.predict(self.train_feed)
        train_auc = metrics.roc_auc_score(self.train_target, train_predict)
        test_predict = self.model.predict(self.test_feed)
        test_auc = metrics.roc_auc_score(self.test_target, test_predict)

        print(train_auc, ' ', test_auc)

    def get_result(self):
        submit_predict = self.model.predict(self.submit_matrix)
        return submit_predict

    def auc_feval(self, preds, xgbtrain):
        label = xgbtrain.get_label()
        score = -metrics.roc_auc_score(label, preds)  # 需要添加负数
        return 'myFeval', score

    def reg_objedt(self):
        pass


if __name__ == '__main__':
    xgb_origin = xgb_model('xgb_origin')
    # rd_forest = rd_forest('rd_forest')
