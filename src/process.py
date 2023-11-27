import os
import sys
import pandas as pd
import logging
import itertools
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.metrics import classification_report, f1_score, r2_score, roc_auc_score, mean_absolute_error,accuracy_score, precision_score, recall_score,mean_squared_error,explained_variance_score
from sklearn import model_selection
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy.stats.stats import pearsonr

def load():
    current_path = os.getcwd()
    try:
        df_train =pd.read_csv(os.path.join(current_path,'src/data/train.txt'), header=None, sep=None)
        df_test=pd.read_csv(os.path.join(current_path,'src/data/test.txt'), header=None, sep=None)
        if len(df_train.columns)==1:
            df_train =pd.read_csv(os.path.join(current_path,'src/data/train.txt'), header=None, delimiter=r"\s+")
            df_test =pd.read_csv(os.path.join(current_path,'src/data/test.txt'), header=None, delimiter=r"\s+")
        print(df_train.head())
    except ValueError:
        print("The filke you have chosen is invalid")
    except FileNotFoundError:
        print(f"No such files of train.txt and test.txt exist")
    return df_train, df_test


def cleaning_raw_df(input_df,symbol=None):
    """ This function check all NAs columns and eliminate them. Furthermore perform interpolation if few NAs for numeric columns
    """
    eliminated = input_df.isnull().all()
    input_df=input_df.loc[:,~eliminated] #eliminate columnes with full NANs
    if symbol:
        for i in symbol:
            for j in input_df.columns.to_list():
                input_df=input_df[input_df[j] != i] #drop elements contained in symbol
    input_df=input_df[input_df.index.notnull()] #eliminate raws withouth any time_data or missing indexing
    numeric_columns = input_df.select_dtypes([np.number]).columns # find the columns that have numeric values (no times)
    input_df[numeric_columns]=input_df[numeric_columns].interpolate(method='linear') #interpolate in by lineear methodology those columns
    if (input_df.index.inferred_type == "datetime64") or (input_df.index.inferred_type == "timedelta"):
        time_columns = input_df.select_dtypes(include=['datetime','timedelta']).columns #find the columns with datetime structure
        input_df[time_columns]=input_df[time_columns].ffill() #use fordward fill.
    input_df = input_df.loc[~input_df.index.duplicated(keep='first')] #eliminate duplicated indexes its kept the first
    return input_df, eliminated

def labeling_df(input_df,label=None, eliminated=None, indexing=None):
    """ read the metadata and asign labels. If columns were elminated in the cleaning process, these columsn are eliminated from the label too.
        ARGS:
        input_df: input data framework
        label: input label without knowing if there are eliminations. Minimum labels are:
            id: if more than one object explaining the output (e.g. engines runing in paralel). Must be numeric
            Indexing: Indexing column that will be used for tracking/rolling information.
            cycles: number of operational times executed a task (can be hours of operations or any number to determinate ttf (if not supplied))
        eliminated: eliminated from clearning_raw_df
        indexing: column that will be used for indexing use labeling information (e.g. 'timecolumn')
    """
    if len(eliminated) != 0:
        label=list(itertools.compress(label,~eliminated))
    if len(label) != 0 :
        input_df.columns = label
    if indexing:
        try:
            input_df[indexing] = pd.to_datetime(input_df[indexing], infer_datetime_format=True)
        except:
            logging.exception("The indexing is not a time based data Or already formated as data")
            pass
        try:
            input_df.set_index(indexing,inplace=True)
        except:
            logging.exception('The indexing cannot be performed. Modify the index column or its data structure')
            exit
    return input_df

def clening_homogenize(training_df,testing_df,remove):
    nonumber_set = training_df.select_dtypes(exclude=['number','bool_']).columns.to_list()
    for i in nonumber_set:
        a=np.sort(training_df[i].unique())
        b=np.sort(testing_df[i].unique())
        if not np.array_equal(a,b):
            mismatch = list(set(b)-set(a))
            if  len(mismatch)<len(a) and len(mismatch)>0:
                testing_df=testing_df[~testing_df[i].isin(mismatch)]
                text=' '.join(map(str,mismatch))
                print(f'in column {i} from the testing set it has been eliminated all raws containg {text}')
            for k in remove:
                training_df[i]=training_df[i].str.replace(k,'')
                testing_df[i]=testing_df[i].str.replace(k,'')
            if (not np.array_equal(np.sort(training_df[i].unique()),np.sort(testing_df[i].unique()))) and len(training_df[i].unique())<=len(testing_df[i].unique()):
                sys.exit('There are difference between the supplied information in the training and testing set that do not allows the execution of the algorithm - ERROR 003')
    return training_df, testing_df, nonumber_set

def preparate_data(input_df,variable=None,period=None,transform_col=None,encoder=None): # period not applied yet.
    """ check if the explicative variable is in place otherwise construct them and add them to the data frame based on the variable named variable
    ARGS:
        input_df: The input data frame
        period: value of the variable at which the class changes for binary classes
        Variable: name of the variable that is linked to the ttf. addictionaly the classification labels are constructed as binary bnc and multiple levesl (3 levels in this case) mcc
    """
    from collections import defaultdict
    #prepare the DF
    if not variable:
        sys.exit('please define the variable that will ERROR - 004')
    if not (('tff' in input_df.columns) or ('label_bnc' in input_df.columns) or ('label_mcc' in input_df.columns)):
        sys.exit('the testing and the training data sets does require to define the explicated column - ttf, label_bnc, or label_mcc ERROR - 005')
    if not transform_col:
        transform_col = input_df.select_dtypes(exclude=['number','bool_']).columns.to_list()
        if 'label_bnc' in input_df  and not 'label_bnc' in transform_col:
            transform_col+=['label_bnc']
        if 'label_mcc' in input_df  and not 'label_mcc' in transform_col:
            transform_col+=['label_mcc']
    encoder_dict = defaultdict(LabelEncoder)
    if not encoder:
        input_df[transform_col] = input_df[transform_col].apply(lambda x: encoder_dict[x.name].fit_transform(x))
    else:
        input_df[transform_col] = input_df[transform_col].apply(lambda x: encoder[x.name].transform(x))
    # to inverse the transformation use lambda too.... input_df[transform_col]=input_df[transform_col].apply(lambda x: encoder_dict[x.name].inverse_transform(x))
    return input_df,encoder_dict,transform_col


def outlier_treatment(input_df,stdcut=None,corrcut=None,id=None,not_processed=None,explicative=None):
    """ remove outliers based on statistical information (i.e. based on individual features standard deviations and correlation - explainability with main variables - ttf)
, and
#the extra arguments are  0: starging feature position 1:ending feature position 2: ttf position. If not supplied assume last two positions are ttf and lable_bcn
    Args:
            input_df: Input dataframe in pandas structure
            stdcut: is the standard devitiation to cut or eliminate a feature
            corrcut: is the correlation cut for the data.
            id: same as before. Define what roaws correspond to a one input machine or component affecting the output ttf. (e.g. several engines working in parallel)
            not_processed = list of string that define what features will NOT be evaluated for outling and will kept, independently. The id and cycles
            explicative = explicative variable (i.e. y variable)
        """
    list_concat=[]
    if id: # id is an identifier for homologues systems.....but want to be considered separated e.g. similar motors or similar countries
        for i in pd.unique(input_df[id]):
            labels_true = [x for x in input_df.columns.to_list() if x not in not_processed]
            df_engine = input_df[input_df[id] == i]
            df_engine_outlier = df_engine[labels_true]
            features_top_var = df_engine_outlier.std()
            features_corr = df_engine_outlier.corrwith(input_df[explicative])
            if stdcut!=None or corrcut!=None:
                listout= np.logical_and(features_top_var,features_corr)
                listout= np.logical_and(listout,features_top_var>stdcut)
                listout= np.logical_and(listout,[True if x>corrcut[1] or x<corrcut[0] else False for x in features_corr])
                if len(list_concat)!=0:
                    list_concat=np.logical_and(list_concat,listout)
                else:
                    list_concat=listout
    else:
        labels_true = [x for x in input_df.columns.to_list() if x not in not_processed]
        df_engine_outlier = input_df[labels_true]
        features_top_var = df_engine_outlier.std()
        features_corr = df_engine_outlier.corrwith(input_df[explicative])
        if stdcut!=None or corrcut!=None:
            listout= np.logical_and(features_top_var,features_corr)
            listout= np.logical_and(listout,features_top_var>stdcut)
            listout= np.logical_and(listout,[True if x>corrcut[1] or x<corrcut[0] else False for x in features_corr])
            if len(list_concat)!=0:
                list_concat=np.logical_and(list_concat,listout)
            else:
                list_concat=listout
    labels = list(itertools.compress(labels_true,list_concat))+not_processed
    output_df=input_df[labels]
    return output_df,labels


class experiments:
    """ Class constructor
    ARGS:
        name: experiment name
        date: experiment runing date
        df_original_tr: original training data frame in pandas
        df_original_ts: original testing data frame in pandas
        df_processed_tr: post cleaning training data frame
        df_processed_ts: post cleaning testing data frame
        normalization: method of normalization
        bin_size: size of the binning
        bin_features: variables that will be binned
        explicative: especification of 'ttf', 'label_bnc', or 'label_mcc'
        sensitive: fairness sensible vairable
        method: object (sklearn) with the parameter for regression or classification.
        model: object (sklearn) after training
        regresive_method:  method used for data augmentation they can be 0,1,2 or 3. 0 is cloning with main variable fair modification, 2 uses the fair correlation, and 3 use the  fariness_aug_methods over the correlated information.   
        x_train: the training explicative set
        y_train: the explained variable of training data set
        x_train_ext: extended features component to be analaysed by metamorphic testing
        y_train_ext: extended features component to be analaysed by metamorphic testing
        x_test: the testing explicative set
        y_test: the testing explained set
        y_hat: prediction of y_test
        y_hat_train: prediction of y_train
        x_keep: internal usage (x data frame kept for extension after oracle observation)
        y_keep: internal usage (y data frame kept for extension after oracle observation)
        metrics_methods: metric method dictionary to be applied for regression or classification estimation
        metrics_values: Resulting metrics
        fairness_metrics_methods: dictionary to be applied for fairness estimation
        fairness_metrics: resulting fairness metrics
        fairness_corr...array: correlation results analyses from fairness states what columns should be kept or eliminated based on different analyses. Check thefunction to understand.
        fairness_corr_df: dataframe that keeps the different correlations
        fairness_aug_method:object (sklearn) with the parameters for regression for estimate non sensitive variables
        fairness_aug_size: limit of rows that can be incorporated in the new structure
        fairness_oracle: dictionary  with the system constrain. The system constraints should follow this structure: {name_rule: 'rule using objects.x_training, or objects.y_training' as variable ]}.*e.g. {'rule1':"objects.x_training['age']>objects.y_training['ttf']" } a > min(b,c) if Rule is None, then is appled only the logic a > b.
        fairness_metamorphic: dictionary with the metamorphic rules
        fairness_matamorphic_results: results from metamorphic testing when doing analysis

        excluded:list of features excluded on the fairness augmentation, independent of correlations
        time_lapse:time lapse of the experiment
        encoder: the encoder used to transform the data
        no_numeric_set: list of the elements (columns) that are not numeric
        random_undersampler: If random undersampling will be performed data augmentation. Algorithmically can be performed before or after.
        omit: column from dataframe to do not consider it in the whole analyses.
    """
    def __init__(self,name,df_original_tr,df_original_ts,explicative,sensitive,bin_size,bin_features,fairness_metamorphic,fairness_metrics_methods,excluded,no_numeric_set,omit,normalization=None,method=None,model=None,regresive_method=None,metrics_methods=None,metrics_values=None,fairness_metric=None,fairness_corr=None,fairness_corr_df=None,fairness_aug_method=None,fairness_aug_size=None,fairness_oracle=None,time_lapse=None,encoder=None,random_undersampler=None):
        self.name = name
        self.df_original_tr = df_original_tr
        self.df_original_ts = df_original_ts
        self.bin_size = bin_size
        self.bin_features = bin_features
        self.explicative = explicative
        self.sensitive = sensitive
        self.fairness_metrics_methods = fairness_metrics_methods
        self.fairness_aug_size = 1000000
        self.fairness_metamorphic = fairness_metamorphic
        self.random_undersampler = False
        self.omit = omit
        self.no_numeric_set=no_numeric_set
        self.excluded = excluded
        self.MULT_TECHNIQUE = ''

        self.df_processed_tr, self.df_processed_ts,self.normalization = [], [], []
        self.method = []
        self.model = []
        self.regresive_method = []
        self.x_train, self.y_train, self.x_train_ext, self.y_train_ext = [], [], [], []
        self.x_test, self.y_test, self.y_hat, self.y_hat_train = [], [], [], []
        self.x_keep, self.y_keep = [], []
        self.metrics_methods, self.metrics_values = [], []
        self.fairness_metric, self.fairness_corr, self.fairness_corr_df, self.fairness_aug_method, self.fairness_oracle, self.fairness_metamorphic_results, self.fairness_metamorphic_metric_results = [], [], [], [], {}, {}, {}
        self.time_lapse = []
        self.encoder = []
        self.no_numeric_set = []

        if normalization is not None:
            self.normalization = normalization
        if method is not None:
            self.method = method
        if model is not None:
            self.model = model
        if regresive_method is not None:
            self.regresive_method = regresive_method
        if metrics_methods is not None:
            self.metrics_methods = metrics_methods
        if metrics_values is not None:
            self.metrics_values = metrics_values
        if fairness_metric is not None:
            self.fairness_metric = fairness_metric
        if fairness_corr is not None:
            self.fairness_corr = fairness_corr
        if fairness_corr_df is not None:
            self.fairness_corr_df = fairness_corr_df
        if fairness_aug_method is not None:
            self.fairness_aug_method = fairness_aug_method
        if fairness_aug_size is not None:
            self.fairness_aug_size = fairness_aug_size
        if fairness_oracle is not None:
            self.fairness_oracle = fairness_oracle
        if time_lapse is not None:
            self.time_lapse = time_lapse
        if encoder is not None:
            self.encoder = encoder
        if random_undersampler is not None:
            self.random_undersampler = random_undersampler

    def metrics(self): # function that estimate the meterics in metrhics_methods if data
        if self.explicative == 'ttf' and len(self.metrics_methods) == 0:
            self.metrics_methods={
                'mse':mean_squared_error(self.y_test,self.y_hat, uniform_average=True,squared=False), #uniform average: average with uniform weight all otputs errors, squared = false return RMSE
                'mae':mean_absolute_error(self.y_test,self.y_hat, uniform_average=True), #uniform average:average with uniform weight all otputs errors
                'r2':r2_score(self.y_test,self.y_hat,uniform_average=True), #uniform average:average with uniform weight all otputs errors
                'explained_variance_score':explained_variance_score(self.y_test,self.y_hat,uniform_average=True)
            }
        elif (self.explicative == 'label_bnc' or self.explicative == 'label_mcc')  and len(self.metrics_methods) == 0:
            self.metrics_methods={
                'Accuracy' : accuracy_score(self.y_test, self.y_hat),
                'Precision' : precision_score(self.y_test, self.y_hat),
                'Recall' : recall_score(self.y_test, self.y_hat),
                'F1 Score' : f1_score(self.y_test,self.y_hat),
                'ROC AUC' : roc_auc_score(self.y_test,self.y_hat),
                       }
        else:
            self.metrics_methods={value:value(self.y_test,self.y_hat) for value in self.metrics_methods.values()}
        df_metrics=pd.DataFrame.from_dict(self.metrics_methods, orient='index')
        df_metrics.columns = [self.method.__class__.__name__+'_RegMeth'+str(self.regresive_method)]
        self.metrics_values=df_metrics
        return self
    def bias(self):
        if len(self.fairness_metrics_methods)!=0:
            output={}
            for key,value in enumerate(self.fairness_metrics_methods):
                output.update({value:eval(self.fairness_metrics_methods[value])})
            df_f_metrics=pd.DataFrame.from_dict(output, orient='index')
            df_f_metrics.columns = [self.method.__class__.__name__+'_RegMeth'+str(self.regresive_method)]
            self.fairness_metric=df_f_metrics
        else:
            self.fairness_metric=[]
        return self
    




import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier, Ridge, LinearRegression, Lasso, ElasticNet,  LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier



def system_preparation(Methods_to_use,regresive_method):
    #load database and clean
    col_names =["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-countrty","label_bnc"]
    explicative = 'label_bnc'
    excluded = ['fnlwgt']
    fairness_oracle = {} #if want to place system constraints
    fairness_metrics_methods = {'fair_1':'pearsonr(self.x_test.sex,self.y_hat)[0]'} #if want to place additional metrics to evaluate
    fairness_metamorphic = {'MR1_type1': [['education','occupation','capital-gain','capital-loss','hours-per-week','workclass'],['sex'],['label_bnc']],'MR2_type1': [['education','occupation','capital-gain','capital-loss','hours-per-week','workclass'],['race'],['label_bnc']],'MR3_type1': [['education','occupation','capital-gain','capital-loss','hours-per-week','workclass'],['age'],['label_bnc']],}
    df_train, df_test = load()
    df_train_clean, eliminated_train = cleaning_raw_df(df_train,symbol=["?"," ?"])
    df_test_clean, eliminated_test = cleaning_raw_df(df_test,symbol=["?"," ?"])
    eliminated=np.logical_and(eliminated_train,eliminated_test)
    df_test_label = labeling_df(df_test_clean,label=col_names,eliminated=eliminated)
    df_train_label = labeling_df(df_train_clean,label=col_names,eliminated=eliminated)
    #homogenize the data structure between test and training data set
    df_train_label,df_test_label, nonumber_set=clening_homogenize(training_df=df_train_label,testing_df=df_test_label,remove=[".",","])
    # preparate the data constructing the main labels if they do not exist and use labelencoders and normalization If too many features, increase the restriction of correlations to eliminate further columns.
    df_train_preparated,labelencoder_train,transform_col=preparate_data(df_train_label, variable=explicative,transform_col=None)
    [df_train_nooutlier,features_corr_train]=outlier_treatment(df_train_preparated,stdcut=0,corrcut=[-0.001,0.001],not_processed=[explicative],explicative=explicative)
    corrcut=0
    while len(df_train_nooutlier.columns) > 50:
        corrcut=corrcut+0.05
        [df_train_nooutlier,features_corr_train]=outlier_treatment(df_train_preparated,stdcut=0,corrcut=corrcut,id='id',not_processed=[explicative],explicative=explicative)
        #normalize the data
    df_test_preparated,labelencoder_test,transform_col=preparate_data(df_test_label, variable=explicative,transform_col=transform_col,encoder=labelencoder_train)
    [df_test_nooutlier,features_corr_test]=outlier_treatment(df_test_preparated,stdcut=0,corrcut=[-0.001,0.001],not_processed=[explicative],explicative=explicative)
    corrcut=0
    while len(df_train_nooutlier.columns) > 50:
        corrcut=corrcut+0.05
        [df_test_nooutlier,features_corr_test]=outlier_treatment(df_test_preparated,stdcut=0,corrcut=corrcut,id='id',not_processed=[explicative],explicative=explicative)
    maxcut=True
    if len(features_corr_train) != len(features_corr_test) and maxcut==True:
        features=[x for x in features_corr_train if x in features_corr_test]
        df_train_nooutlier=df_train_nooutlier[features]
        df_test_nooutlier=df_test_nooutlier[features]
    elif len(features_corr_train) != len(features_corr_test) and maxcut==False:
        df_test_nooutlier=df_test_preparated[features_corr_train]

    # Set objects for  run experiment here is done data curation
    num_exp=len(Methods_to_use)*len(regresive_method)
    objs = [experiments(
            name='Adult'+'_'+str(i)+'_'+explicative,
            df_original_tr=df_train_nooutlier,
            df_original_ts=df_test_nooutlier,
            explicative=explicative,
            excluded=excluded,
            omit=[],
            bin_size=8,
            bin_features=['age'],
            encoder=labelencoder_train,
            no_numeric_set=nonumber_set,
            sensitive=["sex","race","age"],
            fairness_oracle=fairness_oracle,
            fairness_metrics_methods=fairness_metrics_methods,
            random_undersampler=False,
            fairness_metamorphic=fairness_metamorphic,
            normalization=None
            )
            for i in range(num_exp)
            ]

    count,count2 = 0,0
    for i in range(len(objs)):
            objs[i].method =Methods_to_use[count]
            objs[i].regresive_method = regresive_method[count2]
            count+=1
            if count >= len(Methods_to_use):
                count2,count=count2+1,0
    return objs


def bining_and_targets(obj):
    """ Function that creates the x_train,x_test,y_train,and y_test variables"""
    if len(obj[0].bin_features)!=0 and obj[0].bin_size!=0 and obj[0].bin_size!=None:
        for i in obj[0].bin_features:
            if i not in obj[0].df_original_tr.columns.tolist() or obj[0].df_original_tr[i].nunique()<=obj[0].bin_size:
                #labelim.configure(text=f'The variable {i} cannot be binned since the variable cannot be found or the number of bins is to large for that variable')
                print(f'The variable {i} cannot be binned since the variable cannot be found or the number of bins is to large for that variable')
                pass
            else:
                bins = pd.qcut(obj[0].df_original_tr[i], obj[0].bin_size, duplicates='drop')
                meds = obj[0].df_original_tr[i].groupby(bins).agg(['median'])
                label = meds['median'].tolist()
                # Ideally we use the median of the actual values in he bin, as a fall back use the mean of the bin boundaries.
                nan_indices = np.argwhere(np.isnan(label))
                for ni in nan_indices: # if there is any error, the labels are generated as the medium point manually.
                    corrected_label = (meds.iloc[ni].index[0].left + meds.iloc[ni].index[0].right) / 2
                    label[int(ni)] = corrected_label

                obj[0].df_original_tr[i] = pd.qcut(obj[0].df_original_tr[i], obj[0].bin_size, duplicates='drop', labels=label)
                obj[0].df_original_tr[i] = pd.to_numeric(obj[0].df_original_tr[i])
                if len(obj[0].df_original_ts) != 0 or obj[0].df_original_ts!=None:
                    obj[0].df_original_ts[i] = pd.qcut(obj[0].df_original_ts[i], obj[0].bin_size, duplicates='drop', labels=label)
                    obj[0].df_original_ts[i] = pd.to_numeric(obj[0].df_original_ts[i])
                #labelim.configure(text=f'The following features have been binned: {" ".join(obj[0].bin_features)} \n The labels are {label}')
                #labelim.configure(text=f'The following features have been binned: {" ".join(obj[0].bin_features)} \n The labels are {label}')
                print(f'The following features have been binned: {" ".join(obj[0].bin_features)} \n The labels are {label}')
# y_train,y_test,x_train,x_test
    obj[0].df_original_tr=obj[0].df_original_tr.reset_index(drop=True)
    if len(obj[0].df_original_ts) == 0 or obj[0].df_original_ts is None:
        explicative = obj[0].df_original_tr.drop(['id',obj[0].explicative],axis=1,errors='ignore')
        explained = obj[0].df_original_tr[obj[0].explicative]
        obj[0].x_train,obj[0].x_test,obj[0].y_train,obj[0].y_test=model_selection.train_test_split(explicative,explained,train_size=0.65,test_size=0.35,random_state=101)
        #labelim.configure(text=f'Since no testing data was provided. The train set was split in 65% train 35% test randomly chosen')
        print(f'Since no testing data was provided. The train set was split in 65% train 35% test randomly chosen')

        for count in range(len(obj)):
            obj[count].x_train=obj[count].x_train.reset_index(drop=True)
            obj[count].y_train=obj[count].y_train.reset_index(drop=True)
            obj[count].x_test=obj[count].x_test.reset_index(drop=True)
            obj[count].y_test=obj[count].y_test.reset_index(drop=True)
            obj[count].df_original_tr=obj[count].x_train.join(obj[count].y_train)
            obj[count].df_original_ts=obj[count].x_test.join(obj[count].y_test)

    else:
        for count in range(len(obj)):
            obj[count].df_original_tr=obj[0].df_original_tr.reset_index(drop=True)
            obj[count].df_original_ts=obj[0].df_original_ts.reset_index(drop=True)
            obj[count].x_train = obj[0].df_original_tr.drop(['id',obj[0].explicative],axis=1,errors='ignore')
            obj[count].x_test = obj[0].df_original_ts.drop(['id',obj[0].explicative],axis=1,errors='ignore')
            obj[count].y_train = obj[0].df_original_tr[obj[0].explicative]
            obj[count].y_test = obj[0].df_original_ts[obj[0].explicative]
    return obj


def feature_normalization(obj):
    """ Function that normalize numeric variables for experiment variables .... x_train, x_test, y_train and y_test
    ARGS:
    obj: list of experimentation objects.
    """
    for count,experiment in enumerate(obj):
        if experiment.normalization != None and bool(experiment.normalization):
            numeric_co=[x for x in experiment.df_original_tr.columns if x not in experiment.no_numeric_set+[experiment.explicative]+experiment.omit]
            x_train=experiment.normalization.fit_transform(experiment.x_train[numeric_co])
            x_test=experiment.normalization.fit_transform(experiment.x_test[numeric_co])
            obj[count].x_train[numeric_co] = pd.DataFrame(data=x_train,index=experiment.x_train[numeric_co].index,columns=experiment.x_train[numeric_co].columns)
            obj[count].x_test[numeric_co] = pd.DataFrame(data=x_test,index=experiment.x_test[numeric_co].index,columns=experiment.x_test[numeric_co].columns)
            if experiment.explicative in numeric_co:
                y_train=experiment.normalization.fit_transform(experiment.y_train[numeric_co])
                y_test=experiment.normalization.fit_transform(experiment.y_test[numeric_co])
                obj[count].y_train[numeric_co] = pd.DataFrame(data=y_train,index=experiment.x_train[numeric_co].index,columns=experiment.x_train[numeric_co].columns)
                obj[count].y_test[numeric_co] = pd.DataFrame(data=y_test,index=experiment.x_train[numeric_co].index,columns=experiment.x_train[numeric_co].columns)
        obj[count].df_processed_tr=experiment.x_train.join(experiment.y_train).reset_index(drop=True)
        obj[count].df_processed_ts=experiment.x_test.join(experiment.y_test).reset_index(drop=True)
        obj[count].y_train=obj[count].y_train.reset_index(drop=True)
        obj[count].x_train=obj[count].x_train.reset_index(drop=True)
        obj[count].y_test=obj[count].y_test.reset_index(drop=True)
        obj[count].x_test=obj[count].x_test.reset_index(drop=True)
    return obj

def linkage_analyses(obj,thresholds):
    """Provides the correlations analyses that exist on the treated input information. This could be done before or after normalization. we are performing it here before.
    """
    x=obj[0].x_train
    y=obj[0].y_train
    explicative_corr = x.corrwith(y)
    analysis_df=pd.DataFrame(data=explicative_corr,columns=['exp'])
    analysis_results=[]
    for test in obj[0].sensitive:
        analysis_df[test+'W_sensitive'] = abs(x.corrwith(x[test])) # correlation with the sentitive variable (possible to touch)
        analysis_df[test+'W_sensit_notW_expl'] = abs(x.corrwith(x[test]).div(analysis_df['exp'])) # correlates with sensitive variable but not with explained (i.e eliminate or transform)
        analysis_df[test+'W_sensit_W_expl'] = abs(x.corrwith(x[test])*analysis_df['exp']) # correlats with sesntive AND explainable variable (i.e. possible to touch but with careful!!!)
        analysis_df[test+'notW_sensit_W_expl'] = abs(explicative_corr.div(analysis_df[test+'W_sensitive'])) #correlates with explicative but not with sensitive (i.e. not touch!!)
    #... extend more analyses for correlations here!!!!!

        #list 1 and 2 is for touch while list 3 is to not touch
        list_0=analysis_df[test+'W_sensitive']>np.percentile(analysis_df[test+'W_sensitive'],thresholds[0]) # correlation with the sentitive variable (possible to touch)
        list_1=analysis_df[test+'W_sensit_notW_expl']>np.percentile(analysis_df[test+'W_sensit_notW_expl'],thresholds[1]) # correlates with sensitive variable but not with explained (i.e eliminate or transform)
        list_2=analysis_df[test+'W_sensit_W_expl']>np.percentile(analysis_df[test+'W_sensit_W_expl'],thresholds[2]) # correlats with sesntive AND explainable variable (i.e. possible to touch but with careful!!!)
        list_n=analysis_df[test+'notW_sensit_W_expl']>np.percentile(analysis_df[test+'notW_sensit_W_expl'],thresholds[3]) #correlates with explicative but not with sensitive (i.e. not touch!!)
        list_3=list_0*~list_n
        list_4=list_1*~list_n
        list_5=list_2*~list_n
        #make sure that the explicative is not included in the analysis
        for j in obj[0].excluded:
            list_0.loc[j]=False #make sure that the explicative is not included in the analysis
            list_1.loc[j]=False #make sure that the explicative is not included in the analysis
            list_2.loc[j]=False #make sure that the explicative is not included in the analysis
            list_3.loc[j]=False #make sure that the explicative is not included in the analysis
            list_4.loc[j]=False #make sure that the explicative is not included in the analysis
            list_5.loc[j]=False
        if len(analysis_results)!=0:
            analysis_results=[np.logical_or(analysis_results[0],list_0),np.logical_or(analysis_results[1],list_1),np.logical_or(analysis_results[2],list_2),np.logical_or(analysis_results[3],list_3),np.logical_or(analysis_results[4],list_4),np.logical_or(analysis_results[5],list_5)]
        else:
            analysis_results=[list_0,list_1,list_2,list_3,list_4,list_5]
    for i in range(len(obj)):
        obj[i].fairness_corr=analysis_results
        obj[i].fairness_corr_df=analysis_df

    # print('*'*80)
    # print('The correlation analysis has the following results:')
    # #print(analysis_df.to_string())
    # plt.pcolor(analysis_df)
    # plt.yticks(np.arange(0.5, len(analysis_df.index), 1), analysis_df.index)
    # plt.xticks(np.arange(0.5, len(analysis_df.columns), 1), analysis_df.columns)
    # plt.show()
    # print('based on the correlation analyses, the and the percentiles, the list elements are:')

    print(f'correlated with sensitive:')
    print(f'{list_0}')
    print('*'*80)

    print(f'correlated with sensitive not with target:')
    print(f'{list_1}')
    print('*'*80)

    print(f'correlated with sensitive with target:')
    print(f'{list_2}')
    print('*'*80)

    print(f'correlated with sensitive not with target:')
    print(f'{list_3}')
    print('*'*80)

    print(f'correlated with sensitive not with target (higher restriction):')
    print(f'{list_4}')
    print('*'*80)

    print(f'correlated with sensitive not with target:')
    print(f'{list_5}')
    print('*'*80)
    return obj

def perf_measure(y_actual, y_hat):
    TP,FP,TN,FN = 0,0,0,0
    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    return(TP, FP, TN, FN)

def metamorphic_evaluation(object,mbd=False, mbc=False, mbe=False):
    if not (mbd or mbc or mbe):
        sys.exit('When performing metamorphic evaulation, you have to define if it will be performed metamorphic bias detection (mbd), creation (mbc) or evaluation (mbe) Place one as True - ERROR 006')

    if mbd:  ########  evaluate sources of bias based on metamorphic roules ##############
        bias0, bias1, bias2, bias3, bias4, bias5, bias6, bias7, bias8, bias9=[],[],[],[],[],[],[],[],[],[]
        for count,test in object[0].fairness_metamorphic.items():
            agg_df=object[0].df_processed_tr.groupby(by=test[0])
            #bias0=[agg_df.get_group(x).index.to_list() for x in agg_df.groups.keys() if len(agg_df.get_group(x))==1] #list of indexes of elements that fullfill rules with only 1 component
            #bias1=[agg_df.get_group(x).index.to_list() for x in agg_df.groups.keys() for j in test[1]  if len(pd.unique(agg_df.get_group(x)[j]))==1 if len(agg_df.get_group(x))>1] #list of index of elements that fullfill rules in which they have only one representative of the sensitive variables
            #bias2=[agg_df.get_group(x).index.to_list() for x in agg_df.groups.keys() if len(agg_df.get_group(x))/mean(size1)<0.05] # list of elements with discrepancy of sensitive variables bige than 95% of elements in lower direction
            #bias3=[agg_df.get_group(x).index.to_list() for x in agg_df.groups.keys() if len(agg_df.get_group(x))/mean(size1)>20]  # list of elements with discrepancy of sensitive variables bige than 95% of elements in upper direction
            bias4=[agg_df.get_group(x).index.to_list() for x in agg_df.groups.keys() for j in test[1]  if len(pd.unique(agg_df.get_group(x)[j]))>1 if len(pd.unique(agg_df.get_group(x)[object[0].explicative]))>1]
            #bias 4 posess the bias..
            number4=[object[0].df_processed_tr.iloc[bias4[x]]['label_bnc'].value_counts()[0]/object[0].df_processed_tr.iloc[bias4[x]]['label_bnc'].value_counts()[1] for x in range(len(bias4))] #this number tells me the number of 0 respect to number of1 for bias4
            for key in agg_df.groups.keys(): # for each group that fullfill restrictions.
                agg_sensitive=agg_df.get_group(key).groupby(by=test[1])
                #bias5=bias5+[agg_sensitive.get_group(x).index.to_list() for x in agg_sensitive.groups.keys() if len(agg_sensitive.get_group(x))==1]
                #bias6=bias6+[agg_sensitive.get_group(x).index.to_list() for x in agg_sensitive.groups.keys() if len(pd.unique(agg_sensitive.get_group(x)[object[0].explicative]))==1] # list of fullfilled group of sensitive variables that have an unique outcome
                #bias7=bias7+[agg_sensitive.get_group(x).index.to_list() for x in agg_sensitive.groups.keys() if len(agg_sensitive.get_group(x))/mean(size2)<0.05] # list of elements with discrepancy of sensitive variables bige than 95% of elements in lower direction
                #bias8=bias8+[agg_sensitive.get_group(x).index.to_list() for x in agg_sensitive.groups.keys() if len(agg_sensitive.get_group(x))/mean(size2)>20]  # list of elements with discrepancy of sensitive variables bige than 95% of elements in upper direction
                #bias9=bias9+[agg_sensitive.get_group(x).index.to_list() for x in agg_sensitive.groups.keys() for s in test[1] if len(pd.unique(agg_sensitive.get_group(x)[s]))>1 if len(pd.unique(agg_sensitive.get_group(x)[object[0].explicative]))>1 if len(s)>1]
            object[0].fairness_metamorphic_results.update({test[1][0]:test+[bias0,bias1,bias2,bias3,bias4,bias5,bias6,bias7,bias8,bias9,number4]})

        for k in range(len(object)):
            object[k].fairness_metamorphic_results=object[0].fairness_metamorphic_results
    elif mbc:
        pass # this has been moved tolist_processing 2

    elif mbe:   ####metamorphic bias evaluation.
        for k,experiment in enumerate(object):
            kpi1,kpi2,kpi3,kpi4=0,0,0,0

            for counter,test in experiment.fairness_metamorphic_results.items():
                if len(test[4])!=0: #for bias 1
                    pass
                if len(test[7])!=0: ###bias4 ----real bias
                    for num,counting in enumerate(test[7]):
                        dataBase=experiment.df_processed_tr.iloc[counting]
                        for j in test[1]:
                            data=dataBase.drop(experiment.explicative,axis=1,errors='ignore')
                            original=experiment.model.predict(data)
                            valores1=np.unique(experiment.df_processed_tr[j])
                            if len(valores1)==2 and (0 in valores1) and (1 in valores1):
                                inverse = ~data[j].astype('bool')
                                data[j] = inverse.astype('int')
                            else:
                                data[j]=np.random.permutation(data[j])
                            modified=experiment.model.predict(data)
                            kpi1+=sum(abs(np.subtract(original,modified)))
                            y_target=dataBase[experiment.explicative]
                            TP2,FP2,TN2,FN2=perf_measure(y_target.values, original)
                            TP3,FP3,TN3,FN3=perf_measure(y_target.values, modified)
                            if experiment.explicative in ['label_bnc','label_mcc']:
                                accuracy2=(TP2+TN2)/(TP2+FP2+FN2+TN2)
                                accuracy3=(TP3+TN3)/(TP3+FP3+FN3+TN3)
                                if accuracy2!=0 and accuracy3!=0:
                                    kpi2+=max(accuracy2/accuracy3,accuracy3/accuracy2)*100-100
                                else:
                                    kpi2+=100000
                            else:
                                error1=np.subtract(y_target,modified)
                                error2=np.subtract(y_target,original)
                                testing=scipy.stats.ttest_ind(error1,error2)
                                if testing.pvalue>0.05 and len(original)>1:
                                    kpi2+=100*testing.pvalue
                            kpi3+=abs(FP2+FN2-FP3-FN3)
                            kpi4+=abs(TP2+TN2-TP3-TN3)
            object[k].fairness_metamorphic_metric_results=[kpi1,kpi2,kpi3,kpi4]
    return object

def list_processing(object,obj):
    number=object.regresive_method-1
    if object.random_undersampler==True: 
        rus=RandomUnderSampler(random_state=40)
        object.x_train,object.y_train = rus.fit_resample(object.x_train,object.y_train)
    if len(object.x_train_ext)==0:
        object.x_train_ext = object.x_train
        object.y_train_ext = object.y_train
        fair_kpi=object.fairness_corr[number]
        if len(fair_kpi)!=0:
            for i in range(len(fair_kpi)):
                values=np.unique(object.x_train[fair_kpi.index[i]])
                if len(values)==2 and fair_kpi[i] and (0 in values) and (1 in values):
                    inverse = ~object.x_train_ext[fair_kpi.index[i]].astype('bool')
                    object.x_train_ext[fair_kpi.index[i]] = inverse.astype('int')
                elif fair_kpi[i]:
                    object.x_train_ext[fair_kpi.index[i]] = np.random.permutation(object.x_train_ext[fair_kpi.index[i]])
        for j in range(len(obj)):
            if obj[j].regresive_method == object.regresive_method:
                obj[j].x_train_ext=object.x_train_ext
                obj[j].y_train_ext=object.y_train_ext
    else:
        pass
    return object,obj



def list_processing2(object,obj):
    if len(object.x_train_ext)==0:
        number=object.regresive_method-7
        fair_kpi=object.fairness_corr[number]
        final_df=pd.DataFrame()
        if len(fair_kpi)!=0:
            processed_1=[]
            processed_2=[]
            processed_4=[]
            processed_5=[]
            for counter,test in object.fairness_metamorphic_results.items(): ### results to use
                ## for bias 1 and 4
                if len(test[4])!=0: #for bias 1
                    for count in pd.unique(list(itertools.chain.from_iterable(list(test[4]+test[7])))):
                        data=object.df_processed_tr.iloc[[count]]
                        for j in fair_kpi.index[fair_kpi]:
                            valores1=np.unique(object.df_processed_tr[j])
                            if len(valores1)==2 and (0 in valores1) and (1 in valores1):
                                inverse = ~object.df_processed_tr[j][count].astype('bool')
                                data[j] = inverse.astype('int')
                            else:
                                lista = [x for x in list(valores1) if x!=data.iloc[0][j]]
                                if len(lista)>len(data):
                                    #data=data.append([data]*(len(lista)-len(data)),ignore_index=True)
                                    data=pd.concat([data,[data]*(len(lista)-len(data))],ignore_index=True)
                                    data[j]=lista
                                else:
                                    data[j]=np.random.choice(list(valores1),len(data))
                        #final_df=final_df.append(data,ignore_index=True)
                        final_df=pd.concat([final_df,data],ignore_index=True)
                if len(test[7])!=0: ###bias4 ----real bias
                    for num,count in enumerate(test[7]):
                        if count not in processed_4:
                            data=object.df_processed_tr.iloc[count]
                            #data=data.append([data]*5) #this number is a parameter....extensino of the modification....if 1 is clonning
                            data=pd.concat([data,data,data,data,data,data])
                            for j in fair_kpi.index[fair_kpi]:
                                valores1=np.unique(object.df_processed_tr[j])
                                data[j]=np.random.choice(valores1,len(data))
                            if test[13][num]>=1:
                                data[object.explicative]=0
                            elif test[13][num]<1:
                                data[object.explicative]=1
                            #final_df=final_df.append(data,ignore_index=True)
                            final_df=pd.concat([final_df,data],ignore_index=True)
                            processed_4+=[count]
            final_df=final_df.reset_index(drop=True)
            object.x_train_ext =final_df.drop(object.explicative,axis=1)
            object.y_train_ext =final_df[object.explicative]
        for j in range(len(obj)):
            if obj[j].regresive_method == object.regresive_method:
                obj[j].x_train_ext=object.x_train_ext
                obj[j].y_train_ext=object.y_train_ext
    return object,obj

def data_augmentation(obj):
    for count,object in enumerate(obj):
        counter=0
        while True:

            if object.regresive_method==0: # do not perform anything .... i.e. blank analysis
                object.x_train_ext = []
                object.y_train_ext = []
            #if object.regresive_method==1: # clone method
                object.MULT_TECHNIQUE = 'None'
            elif object.regresive_method in [1,2,3,4,5,6]:
                object,obj=list_processing(object,obj)
                object.MULT_TECHNIQUE = 'Cloning'
            elif object.regresive_method in [7,8,9,10,11,12]: # use of metamorphic testing to define wat rows need to be added.
                object,obj=list_processing2(object,obj)
                object.MULT_TECHNIQUE = 'Metamorphic Testing'
            elif object.regresive_method==13: #
                object.x_train_ext = []
                object.y_train_ext = []
            elif object.regresive_method==14: #
                object.x_train_ext = []
                object.y_train_ext = []
            counter+=1
            min_cut=10
            ################# check rules of data augmentation #############
            if bool(object.fairness_oracle) and len(object.x_train_ext)!=0 and len(object.fairness_oracle)!=0:
                for key,value in enumerate(object.fairness_oracle):
                    rule=eval(object.fairness_oracle[value]).to_list()    ### here are incorporated previous knoledge int the system
                    object.x_train_ext=object.x_train_ext[rule]
                    object.y_train_ext=object.y_train_ext[rule]
                    if len(object.x_keep)>0:
                        object.x_train_ext=pd.concat([object.x_train_ext,object.x_keep],ignore_index=True)
                        object.y_train_ext=pd.concat([object.y_train_ext,object.y_keep],ignore_index=True)
            ############# check size of data augmentation ############## need to add a lower cut....
            if len(object.x_train_ext)<=object.fairness_aug_size and len(object.x_train_ext)>min_cut:
                stop=True
            elif len(object.x_train_ext)>object.fairness_aug_size:
                object.x_train_ext.sample(n=object.fairness_aug_size)
                stop=True
            elif len(object.x_train_ext)==0:
                stop=True
                merg_x_train=object.x_train
                merg_y_train=object.y_train
            else:
                object.x_keep=pd.concat([object.x_train_ext,object.x_keep],ignore_index=True)
                object.y_keep=pd.concat([object.y_train_ext,object.y_keep],ignore_index=True)
                merg_x_train=object.x_keep
                merg_y_train=object.y_keep
                stop=False
            if stop == True and len(object.x_train_ext)!=0:
                    merg_x_train=pd.concat([object.x_train,object.x_train_ext],ignore_index=True)
                    merg_y_train=pd.concat([object.y_train,object.y_train_ext],ignore_index=True)
            elif counter ==10: # was not possible to construct
                stop==True
                merg_x_train=object.x_train
                merg_y_train=object.y_train
            if stop:
                obj[count].x_train=merg_x_train
                obj[count].y_train=merg_y_train
                obj[count].df_processed_tr_ext=obj[count].x_train.join(obj[count].y_train).reset_index(drop=True)
                break
    return obj

def run_experiments(obj):
    for cont,exp in enumerate(obj):
        x=exp.method
        method_name = x.__class__.__name__
        try:
            mod=x.fit(exp.x_train, exp.y_train)
            prediction = mod.predict(exp.x_test)
            train_predict =mod.predict(exp.x_train)
            if len(prediction)!= len(exp.y_test):
                sys.exit("The length of the prediction do not match the test set - ERROR 009")
            if len(train_predict)!=len(exp.y_train):
                sys.exit("The length of the trining set do not match the output variable - ERROR 010")

            obj[cont].y_hat=prediction
            obj[cont].y_hat_train=train_predict
            obj[cont].model=mod
            obj[cont].metrics() #call for the metrics estimation
            obj[cont].bias() #call for the bias metrics estimation

            print('*' * 80)
            print(f'Training {method_name} with {exp.explicative} as explicative variable and using the normalization {exp.normalization} and using the {exp.MULT_TECHNIQUE} for data augmentation')
            for i in range(len(obj[cont].metrics_values)):
                print(f'Best Score for {obj[cont].metrics_values.index[i]} is: {obj[cont].metrics_values.iloc[i,0]}')
        except Exception as e:
                print(f'FAILED: {method_name} with transform: {exp.normalization} experiment name: {exp.name} exapetion is {e}')
                pass
    return obj


def show_and_tell(obj):
    try:
        for cont,experiment in enumerate(obj):
            if cont==0:
                dfmix_1=experiment.metrics_values
                dfmix_2=experiment.fairness_metric
                dfmix_3=pd.DataFrame(experiment.fairness_metamorphic_metric_results,columns=experiment.metrics_values.columns)
            else:
                dfmix_1=dfmix_1.join(experiment.metrics_values)
                dfmix_2=dfmix_2.join(experiment.fairness_metric)
                dfmix_3=dfmix_3.join(pd.DataFrame(experiment.fairness_metamorphic_metric_results,columns=experiment.metrics_values.columns))
            print('*********************************')
            print('*********************************')
            print(f'These results  correspond to the process using {experiment.explicative} as explicative variable, {experiment.normalization} as normalization, and {experiment.MULT_TECHNIQUE} as augmentation techinque')
            print('the resulting validtion indexes are:')
            print(f'{experiment.metrics_values}')
            print('*********************************')
            print('The resulting KPIs as model predictors are:')
            print(f'{experiment.fairness_metamorphic_metric_results}')
            print('*********************************')
            print('The resulting KPIs for correlation values is:')
            print(f'{experiment.fairness_metric}')
            print('*********************************')
        print('******************* end of run  *************************')
        dfmix_1=dfmix_1.T.sort_index(axis=0)
        dfmix_2=dfmix_2.T.sort_index(axis=0)
        dfmix_3=dfmix_3.T.sort_index(axis=0)
        dfmix_3[4]=dfmix_3.iloc[:][0]+dfmix_3.iloc[:][1]+dfmix_3.iloc[:][2]+dfmix_3.iloc[:][3]
        #dfmix_3[4][0]+=20
        fig1=dfmix_1.reset_index().plot.bar(x='index',y=dfmix_1.columns.values)
        fig2=dfmix_2.reset_index().plot.bar(x='index',y=dfmix_2.columns.values)
        fig3=dfmix_3.reset_index().plot.bar(x='index',y=dfmix_3.columns.values)
        fig1.savefig('fig1.eps',format='eps')
        fig2.savefig('fig2.eps',format='eps')
        fig3.savefig('fig3.eps',format='eps')
    except:
        pass
    return obj
