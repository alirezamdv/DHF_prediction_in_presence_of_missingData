import pickle
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from backwardAndForward_selection import backwardSelection, forwardSelection
import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

sns.set_style('whitegrid')
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import scale
from sklearn.model_selection import ShuffleSplit, StratifiedKFold
from scipy import stats
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc

# pgmpy
from pgmpy.readwrite.XMLBeliefNetwork import XBNWriter
from pgmpy.readwrite import PomdpXWriter
from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader, XBNReader
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch, HillClimbSearch, BicScore, BDeuScore
from pgmpy.estimators import K2Score, TreeSearch, BayesianEstimator, ParameterEstimator
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


def discretize_data(df, column_name: str, label: str, max_bin=9, min_result=26):
    from ivpy import discretize
    res = discretize(df[column_name], df[label], maxbin=max_bin, minres=min_result)
    return res['breaks'], pd.cut(df[column_name], res['breaks'])


def drop_high_corrs(df, t=0.9):
    # Create correlation matrix
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than t=0.95
    to_drop = [column for column in upper.columns if any(upper[column] > t)]
    # Drop features
    return df.drop(df[to_drop], axis=1)


def read_data(path="day0_discrete_filtered_features.csv", day=-1, sep=","):
    # read data
    d = pd.read_csv(path, sep)
    d = d[d['day0_in'] <= 0]
    df = d.drop_duplicates(subset=['id'], keep='first')
    df = df.reset_index(drop=True)
    # unwanted = ['d01_drowsiness_day', 'd01_retro_day', 'd01_vomiting_day', 'd01_nose_day', 'd01_anorexia_day',
    #             'd01_febrile',
    #             'd01_rashwith_day', 'd01_nausea_day', 'd01_myalgia_day', 'd01_stool_day', 'd01_gum_day',
    #             'd01_diarrhea_day',
    #             'd01_non_prod_day', 'd01_exud_day', 'd01_clear_day', 'd01_bone_day', 'd01_headache_day',
    #             'd01_spont_day', 'd01_feeling_day',
    #             'd01_flushed_day',
    #             'd01_abdominal_day', 'd04_epitro', 'd04_inguinal', 'd04_cervical', 'd04_epitro_d12', 'd04_cervical_d12',
    #             'd04_inguinal_d12', 'd04_epitro_3', 'd04_inguinal_d', 'd04_epitro_da', 'd04_ab_circum_d23',
    #             'd04_cervical_da',
    #             'd04_ab_circum_3', 'd04_cervical_ds', 'd04_epitro_1', 'd04_cervical_d13', 'd04_cervical_2',
    #             'd04_inguinal_1',
    #             'd04_inguinal_d23', 'd04_inguinal_da', 'd04_epitro_d23', 'd04_inguinal_ds', 'd04_cervical_1',
    #             'd04_cervical_d23',
    #             'd04_cervical_3', 'd04_epitro_d13', 'd04_cervical_d', 'd04_epitro_2', 'd04_epitro_ds', 'd04_epitro_d',
    #             'd04_inguinal_d13',
    #             'd04_inguinal_3', 'd04_inguinal_2', 'final_audit_final_dx', 'dengue_dx', 'day0_in', 'id']

    df["d05_ast_alt"] = df['d05_lft_ast'] / df['d05_lft_alt']
    df = df.drop("dengue_dx", axis=1)
    # listd = list(df)
    # ind = [ele for ele in listd if ele not in unwanted]
    # data["d05_ast_alt_1"] = data['d05_lft_ast_1'] / data['d05_lft_alt_1']
    # data["d05_ast_alt_2"] = data['d05_lft_ast_2'] / data['d05_lft_alt_2']
    # data["d05_ast_alt_d"] = data['d05_lft_ast_2'] - data['d05_lft_alt_1']

    # day 3
    # data["d05_ast_alt_1"] = data['d05_lft_ast_1'] / data['d05_lft_alt_1']
    # data["d05_ast_alt_2"] = data['d05_lft_ast_2'] / data['d05_lft_alt_2']
    # data["d05_ast_alt_3"] = data['d05_lft_ast_3'] / data['d05_lft_alt_3']
    # data["d05_ast_alt_m"] = (data['d05_ast_alt_3'] + data['d05_ast_alt_2'] + data['d05_ast_alt_1']) / 3
    # data["d05_ast_alt_d"] = data['d05_lft_ast_2'] - data['d05_lft_ast_3'] - data['d05_lft_alt_1']
    # data["d05_lft_alt_d"] = data['d05_lft_alt_3'] - data['d05_lft_alt_2'] - data['d05_lft_alt_1']
    # data["d04_abdominal_d"] = data['d04_abdominal_3'] - data['d04_abdominal_2'] - data['d04_abdominal_1']
    #
    # data["d05_lymp_d"] = data['d05_lymp_3'] - data['d05_lymp_2'] - data['d05_lymp_1']
    # data["d03_hct_avg_d"] = data['d03_hct_avg_3'] - data['d03_hct_avg_2'] - data['d03_hct_avg_1']
    # data["d05_lft_protein_d"] = data['d05_lft_protein_3'] - data['d05_lft_protein_2'] - data['d05_lft_protein_1']
    # data["d04_ab_circum_d"] = data['d04_ab_circum_3'] - data['d04_ab_circum_2'] - data['d04_ab_circum_1']
    # data["d03_pulse_pre_min_d"] = data['d03_pulse_pre_min_3'] - data['d03_pulse_pre_min_2'] - data[
    #     'd03_pulse_pre_min_1']
    #
    # data["d04_ab_circum_m"] = (data['d04_ab_circum_3'] + data['d04_ab_circum_2'] + data['d04_ab_circum_1']) / 3
    #
    # data["d03_temp_min_d"] = data['d03_temp_min_3'] - data['d03_temp_min_2'] - data['d03_temp_min_1']
    # data["d03_temp_min_m"] = (data['d03_temp_min_3'] + data['d03_temp_min_2'] + data['d03_temp_min_1']) / 3
    # data["d03_pulse_pre_min_m"] = (data['d03_pulse_pre_min_3'] + data['d03_pulse_pre_min_2'] + data[
    #     'd03_pulse_pre_min_1']) / 3

    return df


def roc_auc(data=[], plot=True):
    from sklearn.metrics import roc_curve, roc_auc_score
    for d in data:
        model_name = d[0]
        label = model_name.split('_')[0]
        Y_test = d[1]
        probs = d[2]
        _auc = roc_auc_score(Y_test, probs)
        print(f'{model_name}: AUROC = %0.3f' % _auc)
        # Calculate ROC curve
        _fpr, _tpr, _ = roc_curve(Y_test, probs)

    #     if plot:
    #         plt.plot(_fpr, _tpr, linestyle='--', label=f'{label} (AUROC = %0.3f)' % _auc)

    #         # Title
    #         plt.title('ROC Plot')
    #         # Axis labels
    #         plt.xlabel('False Positive Rate')
    #         plt.ylabel('True Positive Rate')
    #         # Show legend
    #         plt.legend()  #
    #         # Show plot
    # plt.savefig(r'figures/{}.png'.format(model_name))
    # plt.show()
    # print(confusion_matrix(Y_test,probs))
    return _auc


def select_features(df, class_label, n=20) -> pd.DataFrame:
    from sklearn import preprocessing
    from collections import defaultdict
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    try:
        del df["id"]
        del df["day"]
        del df["final_audit_final_dx"]
    except:
        pass
    d = defaultdict(preprocessing.LabelEncoder)
    df = data.apply(lambda x: d[x.name].fit_transform(x))
    y = df[class_label]
    X = df[[f for f in df.columns if f != class_label]]

    # apply SelectKBest class to extract top 10 best features
    best_features = SelectKBest(score_func=chi2, k=10)
    fit = best_features.fit(X, y)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(X.columns)
    # concat two dataframes for better visualization
    feature_scores = pd.concat([df_columns, df_scores], axis=1)
    feature_scores.columns = ['Features', 'Score']  # naming the dataframe columns
    #     print(feature_scores[feature_scores['Features'].isin(FEATURES)].sort_values(by='Score',ascending=False))
    # print(feature_scores.sort_values(by=["Score"], ascending=False).head(50))
    return (feature_scores.nlargest(n, 'Score'))


def check_continues_features(d):
    d = d.copy()
    cl = []
    for i in d.columns:
        num = d[i].nunique()
        if num >= 4:
            #             print("\nColumn Name:",i,"-->",df[i].unique(),"-->Unique Count",len(df[i].unique()),"\n")
            cl.append(i)
    return cl


def check_classifier(x_test, model):
    return model.predict(x_test)


def check_Network(x_test, model, f=["age", "d01_sex"]):
    infer = VariableElimination(model)
    target = "dhf_dx"
    x_test2 = x_test[f]
    result = []
    for c, r in x_test2.iterrows():
        ev = r.to_dict()
        re = infer.query(variables=[target], evidence=ev).values
        result.append(re[1])
    return np.array(result)


def feature_elimination(X, ix):
    if ix:
        return X[ix], X[:ix]
    return X, X


def edge_builder(features, parent="dhf_dx", container=[]):
    res = container.copy()
    for f in features:
        if (f != parent and f != "dhf_dx"):
            res.append((parent, f))
    return res


def all_features(df):
    # day2
    # feature_list = list({'dhf_dx', 'd03_temp_min_2',
    #                      'd04_ab_circum_1',
    #                      'd05_lymp_2',
    #                      'd03_hct_avg_d',
    #                      'd05_lft_ast_1',
    #                      'd05_lft_alt_1',
    #                      'd05_lft_protein_d',
    #                      'd05_lymp_d',
    #                      'd05_lymp_1',
    #                      'd03_temp_min_1',
    #                      'd04_clear_1',
    #                      'd05_lft_protein_2',
    #                      'd03_pulse_rate_range_2',
    #                      'd05_band_1',
    #                      'd05_pmn_d',
    #                      'd05_pmn_2',
    #                      'd04_abdominal_2',
    #                      'd05_atyp_2',
    #                      'd03_hct_avg_2',
    #                      'd04_injected_2',
    #                      'd03_intake_diff_1',
    #                      'd05_ast_alt_1',
    #                      'd04_ab_circum_d',
    #                      'd03_intake_diff_d',
    #                      'd03_intake_diff_2',
    #                      'd04_diarrhea_1',
    #                      'd03_temp_min_d',
    #                      'fever_day',
    #                      'd03_pulse_rate_range_1',
    #                      'd05_pmn_1',
    #                      'd04_itching_2',
    #                      'd05_lft_protein_1',
    #                      'd05_eosin_1',
    #                      'd05_eosin_2',
    #                      'd04_abdominal_1',
    #                      'd05_lft_alt_d',
    #                      'd03_pulse_pre_min_2',
    #                      'd03_pulse_pre_min_d',
    #                      'd03_pulse_pre_min_1',
    #                      'd04_injected_1',
    #                      'd05_ast_alt_d'})
    # day 1

    # feature_list =['d04_lymph', 'd05_ast_alt', 'd05_pmn', 'd01_vomit_bleed_day', 'dhf_dx',
    #      'd05_lft_albumin', 'fever_day', 'd05_band', 'd01_gum_day',
    #      'd01_exud_day', 'd03_pulse_rate_range', 'd03_temp_avg',
    #      'd05_lft_protein', 'd05_lymp', 'd03_intake_diff', 'd03_temp_range',
    #      'age', 'd03_pulse_rate_avg', 'd03_hct_avg', 'd04_itching',
    #      'd05_platelet', 'd04_abdominal', 'd03_temp_max', 'd04_diarrhea',
    #      'd04_maculo', 'd04_limbus', 'd05_atyp', 'd04_mental',
    #      'd01_abdominal_day', 'd05_eosin', 'd04_ab_circum', 'd03_temp_min',
    #      'd01_myalgia_day', 'd05_lft_ast', 'd04_clear']
    feature_list = ['dhf_dx', 'd04_diarrhea', 'd04_abdominal', 'd04_itching', 'd05_atyp',
                    'd04_injected', 'd01_myalgia_day', 'd05_band', 'd03_intake_diff',
                    'd04_clear', 'd05_eosin', 'd03_pulse_rate_range', 'd03_hct_avg',
                    'd05_lymp', 'd05_pmn', 'd03_temp_min', 'd05_lft_protein', 'd05_ast_alt',
                    'd05_sgot_platelet_ratio', 'd04_ab_circum', 'fever_day', 'd03_pulse_pre_min', 'dhf_dx', 'd05_lymp',
                    'd05_lft_protein',
                    'd04_clear', 'd01_height', 'd03_temp_range', 'd03_pulse_rate_avg',
                    'd04_itching', 'd05_pmn', 'd05_platelet', 'age', 'd04_abdominal',
                    'd04_maculo', 'd05_atyp', 'd01_abdominal_day', 'd03_hct_max', 'd04_mental', 'd01_exud_day',
                    'd04_diarrhea', 'd05_band', 'd03_temp_avg', 'd04_limbus',
                    'd04_quantity', 'd01_vomit_bleed_day', 'd01_prod_day', 'd01_gum_day', 'd05_mono', 'd01_myalgia_day',
                    'd04_ab_circum',
                    'd03_intake_diff', 'd05_eosin', 'd03_temp_min', 'd01_feeling_day',
                    'fever_day', 'd03_pulse_rate_range', 'd05_ast_alt', 'd04_lymph', ]

    # day 3
    # feature_list = list({'d03_temp_min_2',
    #                      'd05_ast_alt_2',
    #                      'fever_day',
    #                      'd05_band_1',
    #                      'd04_ab_circum_d',
    #                      'd05_lft_ast_2',
    #                      'd05_lft_ast_1',
    #                      'd04_ab_circum_2',
    #                      'd05_lft_alt_2',
    #                      'd03_temp_min_m',
    #                      'd05_lft_alt_1',
    #                      'd03_pulse_pre_min_2',
    #                      'd04_diarrhea_2',
    #                      'd04_ab_circum_3',
    #                      'd05_eosin_2',
    #                      'd05_lft_alt_d',
    #                      'd03_hct_avg_1',
    #                      'd05_band_2',
    #                      'd05_ast_alt_d',
    #                      'd05_eosin_3',
    #                      'd05_ast_alt_1',
    #                      'd05_lymp_d',
    #                      'd05_pmn_3',
    #                      'd03_pulse_pre_min_d',
    #                      'd04_injected_1',
    #                      'd04_abdominal_1',
    #                      'd04_itching_1',
    #                      'd05_eosin_1',
    #                      'd05_pmn_1',
    #                      'd04_clear_1',
    #                      'd03_pulse_pre_min_1',
    #                      'd03_pulse_rate_range_3',
    #                      'd04_abdominal_2',
    #                      'd03_pulse_rate_range_1',
    #                      'd05_atyp_2',
    #                      'd05_lft_protein_d',
    #                      'd03_temp_min_d',
    #                      'd05_ast_alt_m',
    #                      'd05_lymp_1',
    #                      'd04_itching_2',
    #                      'd04_itching_3',
    #                      'd04_injected_2',
    #                      'd04_diarrhea_1',
    #                      'd03_pulse_pre_min_3',
    #                      'd04_diarrhea_3',
    #                      'd04_injected_3',
    #                      'd03_temp_min_1',
    #                      'd05_lft_protein_1',
    #                      'd05_pmn_2',
    #                      'd05_band_3',
    #                      'd03_pulse_rate_range_2',
    #                      'd03_hct_avg_d',
    #                      'd03_hct_avg_2',
    #                      'd03_intake_diff_2',
    #                      'd03_intake_diff_1',
    #                      'd05_lft_protein_2',
    #                      'd04_ab_circum_1',
    #                      'd04_abdominal_d',
    #                      'd03_temp_min_3',
    #                      'd05_lymp_2',
    #                      'd05_ast_alt_3',
    #                      'd03_hct_avg_3',
    #                      'd04_abdominal_3',
    #                      'd05_lymp_3',
    #                      'd05_lft_protein_3',
    #                      'd03_intake_diff_3', 'dhf_dx'
    #                      })
    feature_list.extend(list(select_features(data, "dhf_dx", n=50)['Features']))
    return list(set(feature_list))
    # return feature_list


def best_edges(day=3):
    if day == 3:
        return [
            ('dhf_dx', 'd04_diarrhea'),
            ('dhf_dx', 'd04_abdominal'),
            ('dhf_dx', 'd04_itching'),
            ('dhf_dx', 'd05_atyp'),
            ('dhf_dx', 'd04_injected'),
            ('dhf_dx', 'd01_myalgia_day'),
            ('dhf_dx', 'd05_band'),
            ('dhf_dx', 'd03_intake_diff'),
            ('dhf_dx', 'd04_clear'),
            ('dhf_dx', 'd05_eosin'),
            ('dhf_dx', 'd03_pulse_rate_range'),
            ('dhf_dx', 'd03_hct_avg'),
            ('dhf_dx', 'd05_lymp'),
            ('dhf_dx', 'd05_pmn'),
            ('dhf_dx', 'd03_temp_min'),
            ('dhf_dx', 'd05_lft_protein'),
            ('dhf_dx', 'd05_ast_alt'),
            ('dhf_dx', 'd05_sgot_platelet_ratio'),
            ('dhf_dx', 'd04_ab_circum'),
            ("dhf_dx", "fever_day"),
            ('fever_day', 'd05_sgot_platelet_ratio'),
            ('fever_day', 'd04_abdominal'),
            ('fever_day', 'd04_injected'),
            ('fever_day', 'd04_itching'),
        ]


def discretizer(dataframe):
    bins = {}
    for c in check_continues_features(dataframe):
        b, d = discretize_data(dataframe, c, "dhf_dx")
        bins[c] = b
        #     print(c, " : ", b)
        dataframe[c] = d
    for i in dataframe.dtypes.keys():
        if dataframe.dtypes[i] == 'category':
            dataframe[i] = dataframe[i].cat.codes
    return dataframe


if __name__ == '__main__':

    report = []
    # features_impacts['model_performance'] = []
    model_performance = []
    data = read_data('../Data/data_final - Ordered.csv')
    features = all_features(data)
    # features += ['dhf_dx']
    # features.remove("d05_sgot_sgpt_ratio")
    # features.remove("dengue_dx")
    # features.remove("d05_alb_globn_ratio")

    data = data[features]
    data = discretizer(data)

    # X = data[[f for f in data.columns if f != "dhf_dx"]]
    # y = data['dhf_dx']
    # ix = None

    # we add an integer to feature list just for getting all features in first step
    features.insert(0, 1)
    # parent = 'd03_temp_min'
    features.remove('dhf_dx')

    noisy_feat = []
    edges = []
    for j, f in enumerate(features):
        df = data.copy()
        print("=" * 20)
        print(f'length of features before elimination: {len(df.columns)}')
        fig, ax = plt.subplots(figsize=(10, 10))
        # this part is for feature selection
        ################################################
        if f in df.columns:
            df = df.drop([f], axis=1)
            print(f'removed {f} length of features : {len(df.columns)}')

        if len(noisy_feat) > 0:
            for noisy in noisy_feat:
                df = df.drop([noisy], axis=1)
                print("=" * 10, f"deleted {noisy} ", len(df.columns), "=" * 10)
        ##########################################################
        # this is for fever_day nodes
        # print("=" * 20)
        #
        # if f in df.columns:
        #     if f != parent and f != "dhf_dx":
        #         ed = edge_builder([f], parent, [])
        #         print(f"added {ed[0]}")
        #         edges.append(ed[0])
        # if len(noisy_feat) > 0:
        #     for noisy in noisy_feat:
        #         try:
        #             edges.remove(noisy)
        #             print("=" * 10, f"deleted {noisy} ", len(edges), "=" * 10)
        #         except:
        #             pass
        print("=" * 20)
        # arcs = edge_builder(df.columns, 'dhf_dx', edges)
        # arcs = best_edges(3)
        # arcs += edges
        # model = BayesianModel(arcs)
        # print("=" * 10, "edge_length", len(model.edges), "=" * 10)

        classifier = CategoricalNB()
        accuracies = []
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)
        name = f'model_{j}'
        rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
        count = 0
        # forward selection in between:
        l = (len(df.columns) - 1)
        X = df.drop(['dhf_dx'], axis=1)
        y = df[['dhf_dx']]
        # sfs = SFS(CategoricalNB(),
        #           k_features=l,
        #           forward=True,
        #           floating=True,
        #           #            scoring='neg_mean_squared_error',
        #           scoring='accuracy',
        #           cv=3)
        # sfs.fit(X, y)
        # df_SFS_results = pd.DataFrame(sfs.subsets_).transpose()
        # df2 = df[list(
        #     df_SFS_results[(df_SFS_results['avg_score'] == df_SFS_results['avg_score'].max())]['feature_names'].values[
        #         0])]
        # df2 = df.iloc[:, list(sfs.k_feature_idx_)]
        # df2["dhf_dx"] = df["dhf_dx"]
        # Backward cross validation:
        for train_index, test_index in rs.split(df):
            X_train = df.iloc[train_index]
            y_train = X_train[["dhf_dx"]]
            X_train = X_train.drop(['dhf_dx'], axis=1)
            x_test = df.iloc[test_index]
            y_test = np.array(x_test[["dhf_dx"]].values)
            X_test = x_test.drop(["dhf_dx"], axis=1)
            classifier.fit(X_train.values, y_train)
            viz = RocCurveDisplay.from_estimator(
                classifier,
                X_test,
                y_test,
                name="ROC fold {}".format(count),
                alpha=0.3,
                lw=1,
                ax=ax,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            accuracies.append(viz.roc_auc)
            count += 1

        # ev_f = [n for n in model.nodes() if n != "dhf_dx"]

        # cross validation:
        # count = 0
        # for train_index, test_index in rs.split(df):
        #     X_train = df.iloc[train_index]
        #     X_test = df.iloc[test_index]
        #     x_test = X_test[[col for col in X_test.columns if col != "dhf_dx"]]
        #     y_test = np.array(X_test[["dhf_dx"]].values)
        #     # training:
        #     model.fit(X_train, estimator=BayesianEstimator)
        #     # res = check_network(X_test, model, ev_f)
        #     # evaluation
        #
        #     writer = XMLBIFWriter(model)
        #     writer.write_xmlbif(f'models/{model_name}.xml')
        #     # writer = XBNWriter(model = model)
        #     t = [[name + '_' + str(count), y_test, res]]
        #     auc = roc_auc(t)
        #     accuracies.append(auc)
        #     count += 1

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        accuracy_mean = auc(mean_fpr, mean_tpr)
        if len(model_performance) > 0:
            max_acc = max(model_performance)
            # features_impacts[name] = max_acc - accuracy_mean
            feature_impact = max_acc - accuracy_mean
            if feature_impact < 0.0:
                noisy_feat.append(f)
            report.append(
                {"name": name, "acc": accuracy_mean, "impact": feature_impact, "featurse": df.columns, 'feature': f,
                 'n': len(df.columns)})
            with open('models/d1_perform.pickle', 'wb') as handle:
                pickle.dump(report, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print(
                '=======================================================================================================')
            print(f'model name: {name} performance {accuracy_mean}, features_length: {len(df.columns)} ,\
              diffrence:{feature_impact} \n , max_acc: {max_acc}')

            print(
                '=======================================================================================================')

        model_performance.append(accuracy_mean)
