import pandas as pd
import numpy as np


def clean_adult(original_dataset_path):

    dataset = pd.read_csv(original_dataset_path)

    def basic_clean():
        #dataset["gender"] = dataset["gender"].map({"Male": 1, "Female":0})
        dataset['income']=dataset['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
        dataset.drop(labels=["fnlwgt", "educational_num", "race", "native_country", "relationship"], axis = 1, inplace = True)    

    def process_education():
        education = []
        for index, row in dataset.iterrows():

            if(row['education'] in ["10th", "11th", "12th", "1st-4th", "5th-6th", "7th-8th", "9th", "Preschool"] ):
                education.append("dropout")

            if(row['education'] in ["Assoc-acdm", "Assoc-voc"] ):
                education.append("associates")

            if(row['education'] in ["Bachelors"] ):
                education.append("bachelors")

            if(row['education'] in ["Masters", "Doctorate"] ):
                education.append("masters_doctorate")

            if(row['education'] in ["HS-grad", "Some-college"] ):
                education.append("hs_grad")

            if(row['education'] in ["Prof-school"] ):
                education.append("prof_school")
            
        dataset['education'] = education

    def process_workclass():
        workclass = []
        for index, row in dataset.iterrows():

            if(row['workclass'] in ["Federal-gov"] ):
                workclass.append("fedGov")

            if(row['workclass'] in ["Local-gov", "State-gov"] ):
                workclass.append("otherGov")

            if(row['workclass'] in ["Private"] ):
                workclass.append("private")

            if(row['workclass'] in ["Self-emp-inc", "Self-emp-not-inc"] ):
                workclass.append("selfEmployed")

            if(row['workclass'] in ["Without-pay", "Never-worked" ] ):
                workclass.append("unEmployed")

        dataset['workclass'] = workclass

    def process_occupation():
        occupation = []
        for index, row in dataset.iterrows():

            if(row['occupation'] in ["Craft-repair", "Farming-fishing","Handlers-cleaners", "Machine-op-inspct", "Transport-moving"] ):
                occupation.append("blueCollar")

            if(row['occupation'] in ["Exec-managerial"] ):
                occupation.append("whiteCollar")
            
            if(row['occupation'] in ["Sales"] ):
                occupation.append("sales")

            if(row['occupation'] in ["Prof-specialty"] ):
                occupation.append("professional")

            if(row['occupation'] in ["Tech-support", "Protective-serv", "Armed-Forces", "Other-service", "Priv-house-serv", "Adm-clerical"] ):
                    occupation.append("other")

        dataset['occupation'] = occupation

    def process_marital():
        marital = []
        for index, row in dataset.iterrows():

            if(row['marital_status'] in ["Never-married"] ):
                marital.append("single")

            if(row['marital_status'] in ["Married-civ-spouse", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse" ] ):
                marital.append("married")


        dataset['marital_status'] = marital


    basic_clean()
    process_education()
    process_workclass()
    process_occupation()
    process_marital()

    return dataset


def clean_compas(original_dataset_path):

    dataset = pd.read_csv(original_dataset_path)

    def basic_clean():
        dataset.drop(labels=["age_cat", "sex-race", "c_charge_desc"], axis = 1, inplace = True)    

    def process_feloniesNB():
        fel_cnt = []
        for index, row in dataset.iterrows():
            if int(row['juv_fel_count']) == 0:
                fel_cnt.append("=0")
            else:
                fel_cnt.append(">0")         
        dataset.drop(labels=["juv_fel_count"], axis = 1, inplace = True)
        dataset["juvenile-felonies"] = fel_cnt
        
    def process_misdemeanorsNB():
        msd_cnt = []
        for index, row in dataset.iterrows():
            if int(row['juv_misd_count']) == 0:
                msd_cnt.append("=0")
            else:
                msd_cnt.append(">0")         
        dataset.drop(labels=["juv_misd_count"], axis = 1, inplace = True)
        dataset["juvenile-misdemeanors"] = msd_cnt

    def process_crimesNB():
        crimes_cnt = []
        for index, row in dataset.iterrows():
            if int(row['juv_other_count']) == 0:
                crimes_cnt.append("=0")
            else:
                crimes_cnt.append(">0")         
        dataset.drop(labels=["juv_other_count"], axis = 1, inplace = True)
        dataset["juvenile-crimes"] = crimes_cnt

    def process_age():
        age_categs = []
        for index, row in dataset.iterrows():
            if 18 <= int(row['age']) and  int(row['age']) <= 20:
                age_categs.append("18-20")
            elif 21 <= int(row['age']) and  int(row['age']) <= 22:
                age_categs.append("21-22")
            elif 23 <= int(row['age']) and  int(row['age']) <= 25:
                age_categs.append("23-25")
            elif 26 <= int(row['age']) and  int(row['age']) <= 45:
                age_categs.append("26-45")
            elif int(row['age']) > 45:
                age_categs.append(">45")
            else:
                age_categs.append("<18")       
        dataset['age'] = age_categs

    def process_priors():
        priors_cnt = []
        for index, row in dataset.iterrows():
            if 2 <= int(row['priors_count']) and  int(row['priors_count']) <= 3:
                priors_cnt.append("2-3")
            elif int(row['priors_count']) == 0:
                priors_cnt.append("0")
            elif int(row['priors_count']) == 1:
                priors_cnt.append("1")
            elif 3 < int(row['priors_count']):
                priors_cnt.append(">3")
            else:
                priors_cnt.append("error")       
        dataset.drop(labels=["priors_count"], axis = 1, inplace = True)
        dataset["priors"] = priors_cnt

    basic_clean()
    process_feloniesNB()
    process_misdemeanorsNB()
    process_crimesNB()
    process_age()
    process_priors()
    return dataset

def clean_german_credit(original_dataset_path):

    dataset = pd.read_csv(original_dataset_path)

    def basic_clean():
        dataset.drop(labels=[], axis = 1, inplace = True)    

    def process_age():
        age_categs = []
        for index, row in dataset.iterrows():
            if 25 > int(row['age']): # First quartile is actually 27
                age_categs.append("_<25")
            elif 25 <= int(row['age']) and  int(row['age']) < 33:
                age_categs.append("25<_<33")
            elif 33 <= int(row['age']) and  int(row['age']) < 42:
                age_categs.append("33<_<42")
            elif 42 <= int(row['age']):
                age_categs.append("42<_")
            else:
                age_categs.append("error")       
        dataset['age'] = age_categs

    def process_credit_duration():
            priors_cnt = []
            for index, row in dataset.iterrows():
                if 12 > int(row['credit_duration_months']):
                    priors_cnt.append("_<12")
                elif (12 <= int(row['credit_duration_months'])) and (int(row['credit_duration_months']) < 18):
                    priors_cnt.append("12<_<18")
                elif (18 <= int(row['credit_duration_months'])) and (int(row['credit_duration_months']) < 24):
                    priors_cnt.append("18<_<24")
                elif 24 <= int(row['credit_duration_months']):
                    priors_cnt.append("24<_")
                else:
                    priors_cnt.append("error")       
            dataset.drop(labels=["credit_duration_months"], axis = 1, inplace = True)
            dataset["credit_duration_months"] = priors_cnt

    def process_amount():
        priors_cnt = []
        for index, row in dataset.iterrows():
            if 1364.5 > int(row['credit_amount']):
                priors_cnt.append("_<1364.5")
            elif (1364.5 <= int(row['credit_amount'])) and (int(row['credit_amount']) < 2319.5):
                priors_cnt.append("1364.5<_<2319.5")
            elif (2319.5 <= int(row['credit_amount'])) and (int(row['credit_amount']) < 3972.75):
                priors_cnt.append("2319.5<_<3972.75")
            elif 3972.75 <= int(row['credit_amount']):
                priors_cnt.append("3972.75<_")
            else:
                priors_cnt.append("error")       
        dataset.drop(labels=["credit_amount"], axis = 1, inplace = True)
        dataset["credit_amount"] = priors_cnt

    def process_installment_rate():
        priors_cnt = []
        for index, row in dataset.iterrows():
            if 1 == int(row['installment_rate(perc_disp_income)']):
                priors_cnt.append("rate_1")
            elif 2 == int(row['installment_rate(perc_disp_income)']):
                priors_cnt.append("rate_2")
            elif 3 == int(row['installment_rate(perc_disp_income)']):
                priors_cnt.append("rate_3")
            elif 4 == int(row['installment_rate(perc_disp_income)']):
                priors_cnt.append("rate_4")
            else:
                priors_cnt.append("error")       
        dataset.drop(labels=["installment_rate(perc_disp_income)"], axis = 1, inplace = True)
        dataset["installment_rate"] = priors_cnt

    def process_bank_credits():
        priors_cnt = []
        for index, row in dataset.iterrows():
            if 1 == int(row['bank_credits']):
                priors_cnt.append("1_")
            elif 2 == int(row['bank_credits']):
                priors_cnt.append("2_")
            elif 3 == int(row['bank_credits']):
                priors_cnt.append("3_")
            elif 4 == int(row['bank_credits']):
                priors_cnt.append("4_")
            else:
                priors_cnt.append("error")       
        dataset.drop(labels=["bank_credits"], axis = 1, inplace = True)
        dataset["bank_existing_credits"] = priors_cnt

    def process_residence_duration():
        priors_cnt = []
        for index, row in dataset.iterrows():
            if 1 == int(row['residence_duration']):
                priors_cnt.append("1_")
            elif 2 == int(row['residence_duration']):
                priors_cnt.append("2_")
            elif 3 == int(row['residence_duration']):
                priors_cnt.append("3_")
            elif 4 == int(row['residence_duration']):
                priors_cnt.append("4_")
            else:
                priors_cnt.append("error")       
        dataset.drop(labels=["residence_duration"], axis = 1, inplace = True)
        dataset["residence_duration"] = priors_cnt

    def process_dependents():
        priors_cnt = []
        for index, row in dataset.iterrows():
            if 1 == int(row['dependents']):
                priors_cnt.append("1_dep")
            elif 2 == int(row['dependents']):
                priors_cnt.append("2_deps")
            else:
                priors_cnt.append("error")       
        dataset.drop(labels=["dependents"], axis = 1, inplace = True)
        dataset["dependents"] = priors_cnt



    basic_clean()
    process_age()
    process_amount()
    process_credit_duration()
    process_installment_rate()
    process_bank_credits()
    process_residence_duration()
    process_dependents()

    return dataset

def clean_default_credit(original_dataset_path):

    dataset = pd.read_csv(original_dataset_path)

    def basic_clean():
        dataset.drop(labels=[], axis = 1, inplace = True)    

    def process_sex():
        sex_categs = []
        for index, row in dataset.iterrows():
            if 1 == int(row['SEX']):
                sex_categs.append("male")
            elif 2 == int(row['SEX']):
                sex_categs.append("female")
            else:
                sex_categs.append("error")   
                print("error : in education, found ", row['SEX'])    
        dataset['SEX'] = sex_categs

    def process_education():
        tot = 0
        educ_categs = []
        for index, row in dataset.iterrows():
            if 1 == int(row['EDUCATION']):
                educ_categs.append("grad_school")
            elif 2 == int(row['EDUCATION']):
                educ_categs.append("university")
            elif 3 == int(row['EDUCATION']):
                educ_categs.append("high_school")
            elif 4 == int(row['EDUCATION']):
                educ_categs.append("others")
            elif 5 == int(row['EDUCATION']) or 6 == int(row['EDUCATION']):
                educ_categs.append("unknown")
            elif 0 == int(row['EDUCATION']):
                educ_categs.append("na")
            else:
                educ_categs.append("error")   
                print("error : in education, found ", row['EDUCATION'])   
                tot = tot + 1
        if tot > 0:
            print("Found ", tot, " errors here") 
        dataset['EDUCATION'] = educ_categs

    def process_marriage():
        marr_categs = []
        tot = 0
        for index, row in dataset.iterrows():
            if 1 == int(row['MARRIAGE']):
                marr_categs.append("married")
            elif 2 == int(row['MARRIAGE']):
                marr_categs.append("single")
            elif 3 == int(row['MARRIAGE']):
                marr_categs.append("others")
            elif 0 == int(row['MARRIAGE']):
                marr_categs.append("na")
            else:
                marr_categs.append("error")    
                tot = tot + 1
                print("error : in marriage, found ", row['MARRIAGE'])   
        if tot > 0:
            print("Found ", tot, " errors here") 
        dataset['MARRIAGE'] = marr_categs

    def process_pay(i):
        attr = "PAY_%d" %i
        pay_bin = []
        if i == 0:
            for index, row in dataset.iterrows():
                if -1 == int(row[attr]):
                    pay_bin.append("pay_duly")
                elif -2 == int(row[attr]):
                    pay_bin.append("no_consumption")
                elif 0 == int(row[attr]):
                    pay_bin.append("revolving_credit")
                elif 1 == int(row[attr]):
                    pay_bin.append("delay_1_month")
                elif 1 < int(row[attr]) and int(row[attr]) <= 9:
                    pay_bin.append("delay_>1_month")
                else:
                    pay_bin.append("error")      
                    print("error : in ", attr, " found ", row[attr])
        else: 
            for index, row in dataset.iterrows():
                if -1 == int(row[attr]):
                    pay_bin.append("pay_duly")
                elif -2 == int(row[attr]):
                    pay_bin.append("no_consumption")
                elif 0 == int(row[attr]):
                    pay_bin.append("revolving_credit")
                elif 1 <= int(row[attr]) and int(row[attr]) <= 9:
                    pay_bin.append("delay_>=1_month")
                else:
                    pay_bin.append("error")   
                    print("error : in ", attr, " found ", row[attr])
        dataset[attr] = pay_bin

    def process_bill_amt(i):
        attr = "BILL_AMT%d" %i
        bill_bin = []
        if i == 2:
            q1 = 2984.5
            q2 = 21200
            q3 = 64008.5
        elif i == 3:
            q1 = 2665.5
            q2 = 20088.5
            q3 = 60165.5
        elif i == 4:
            q1 = 2326.5
            q2 = 19052
            q3 = 54509
        elif i == 5:
            q1 = 1763
            q2 = 18104.5
            q3 = 50196
        elif i == 6:
            q1 = 1256
            q2 = 17071
            q3 = 49200.5
        for index, row in dataset.iterrows():
            if int(row[attr]) <= q1:
                bill_bin.append("<=%d" %q1)
            elif q1 < int(row[attr]) and int(row[attr]) <= q2:
                bill_bin.append("%d<_<=%d" %(q1,q2))
            elif q2 < int(row[attr]) and int(row[attr]) <= q3:
                bill_bin.append("%d<_<=%d" %(q2,q3))
            elif q3 < int(row[attr]):
                bill_bin.append(">%d" %q3)
            else:
                bill_bin.append("error")    
                print("error : in ", attr, " found ", row[attr])   
        dataset[attr] = bill_bin

    basic_clean()
    process_sex()
    process_education()
    process_marriage()
    for i in range(0,7):
        if i != 1:
            process_pay(i)
    for j in range(2,7):
        process_bill_amt(j)
    return dataset