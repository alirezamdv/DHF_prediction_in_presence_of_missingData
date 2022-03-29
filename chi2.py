
def read_data(path="day0_discrete_filtered_features.csv",day=-1 , sep=","):
    #read data
    df = pd.read_csv(path,sep)
    if day>=0:
        prop = (df.day==day)
        df=df[prop]
    return df
    
    


def select_features(df,class_label,n=20)->pd.DataFrame:
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
    y=df[class_label]
    X = df[[f for f in df.columns if f != class_label]]

    # apply SelectKBest class to extract top 10 best features
    best_features = SelectKBest(score_func=chi2, k=10)
    fit = best_features.fit(X,y)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    feature_scores = pd.concat([df_columns,df_scores],axis=1)
    feature_scores.columns = ['Features','Score']  #naming the dataframe columns
#     print(feature_scores[feature_scores['Features'].isin(FEATURES)].sort_values(by='Score',ascending=False))
#     print(feature_scores.sort_values(by=["Score"],ascending=False).head(50))
    return(feature_scores.nlargest(n,'Score')) 



if __name__ == '__main__':
    data = read_data("data_final - Ordered.csv",day=-1)
    d = data[data['day0_in']<=0]
    data = d.drop_duplicates(subset=['id'],keep='first')
    data["d05_ast_alt"] = data['d05_lft_ast'] / data['d05_lft_alt']
    data = data.reset_index(drop=True)

    features = list(select_features(data,"dhf_dx",n=50)['Features'])
    print(features)