from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/prediksi", methods=['POST'])
def prediksi():
    #room = float(request.form['room'])
    # --------------------------------------------------------------------------------------------------------------------
    # loan_amnt = [5000.0, 2500.0]
    # int_rate = [10.65, 15.27]
    # installment = [162.87, 59.83]
    # annual_inc = [24000.0, 30000.0]
    # dti = [27.65, 1.00]
    # delinq_2yrs = [0.0, 0.0]
    # inq_last_6mths = [1.0, 5.0]
    # total_pymnt = [5861.071414, 1008.710000]
    # total_rec_prncp = [5000.0, 456.46]
    # recoveries = [0.0, 117.08]
    # collection_recovery_fee = [0.0, 1.11]
    # last_pymnt_amnt = [171.62, 119.66]
    # issue_d = ['2011-12-1', '2011-12-1']
    # last_pymnt_d = ['2015-1-1', '2013-4-1']
    # last_credit_pull_d = ['2016-1-1', '2013-9-1']

    loan_amnt = float(request.form['loan_amnt'])
    int_rate = float(request.form['int_rate'])
    installment = float(request.form['installment'])
    annual_inc = float(request.form['annual_inc'])
    dti = float(request.form['dti'])
    delinq_2yrs = float(request.form['delinq_2yrs'])
    inq_last_6mths = float(request.form['inq_last_6mths'])
    total_pymnt = float(request.form['total_pymnt'])
    total_rec_prncp = float(request.form['total_rec_prncp'])
    recoveries = float(request.form['recoveries'])
    collection_recovery_fee = float(request.form['collection_recovery_fee'])
    last_pymnt_amnt = float(request.form['last_pymnt_amnt'])

    issue_d = request.form['issue_d'].split('-')
    issue_d_y, issue_d_m, issue_d_d = int(
        issue_d[0]), int(issue_d[1]), int(issue_d[2])
    issue_d = pd.DataFrame({'year': [issue_d_y], 'month': [
                           issue_d_m], 'day': [issue_d_d]})
    issue_d = pd.to_datetime(issue_d).astype(np.int64)*1.0

    last_pymnt_d = request.form['last_pymnt_d'].split('-')
    last_pymnt_d_y, last_pymnt_d_m, last_pymnt_d_d = int(
        last_pymnt_d[0]), int(last_pymnt_d[1]), int(last_pymnt_d[2])
    last_pymnt_d = pd.DataFrame({'year': [last_pymnt_d_y], 'month': [
                                last_pymnt_d_m], 'day': [last_pymnt_d_d]})
    last_pymnt_d = pd.to_datetime(last_pymnt_d).astype(np.int64)*1.0

    last_credit_pull_d = request.form['last_credit_pull_d'].split('-')
    last_credit_pull_d_y, last_credit_pull_d_m, last_credit_pull_d_d = int(
        last_credit_pull_d[0]), int(last_credit_pull_d[1]), int(last_credit_pull_d[2])
    last_credit_pull_d = pd.DataFrame({'year': [last_credit_pull_d_y], 'month': [
                                      last_credit_pull_d_m], 'day': [last_credit_pull_d_d]})
    last_credit_pull_d = pd.to_datetime(
        last_credit_pull_d).astype(np.int64)*1.0

    fitur_numerik_tes = {
        'loan_amnt': pd.Series(loan_amnt),
        'int_rate': int_rate,
        'installment': installment,
        'annual_inc': annual_inc,
        'dti': dti,
        'delinq_2yrs': delinq_2yrs,
        'inq_last_6mths': inq_last_6mths,
        'total_pymnt': total_pymnt,
        'total_rec_prncp': total_rec_prncp,
        'recoveries': recoveries,
        'collection_recovery_fee': collection_recovery_fee,
        'last_pymnt_amnt': last_pymnt_amnt,
        'issue_d': issue_d,
        'last_pymnt_d': last_pymnt_d,
        'last_credit_pull_d': last_credit_pull_d
    }

    df_numerik_tes = pd.DataFrame()
    for i in fitur_numerik_tes:
        df_numerik_tes[i] = fitur_numerik_tes[i]
    # --------------------------------------------------------------------------------------------------------------------
    # term = ['36 months', '60 months']
    # grade = ['B', 'C']
    # sub_grade = ['B2', 'C4']
    # emp_length = [10.0, 0.0]
    # home_ownership = ['RENT', 'RENT']
    # purpose = ['credit_card', 'car']

    term = request.form['term']
    grade = request.form['grade']
    sub_grade = request.form['sub_grade']
    emp_length = float(request.form['emp_length'])
    home_ownership = request.form['home_ownership']
    purpose = request.form['purpose']

    fitur_kategorik_tes = {
        'daftar_term': ['36 months'],
        'daftar_grade': ['A', 'B', 'C', 'D', 'E', 'F'],
        'daftar_sub_grade': ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4'],
        'daftar_emp_length': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        'daftar_home_ownership': ['ANY', 'MORTGAGE', 'NONE', 'OTHER', 'OWN'],
        'daftar_purpose': ['car', 'credit_card', 'debt_consolidation', 'educational', 'home_improvement', 'house', 'major_purchase', 'medical', 'moving', 'other', 'renewable_energy', 'small_business', 'vacation']
    }

    df_kategorik_tes = pd.DataFrame()
    for i in fitur_kategorik_tes:
        for j in fitur_kategorik_tes[i]:
            df_kategorik_tes[j] = pd.Series(float((j == term) |
                                                  (j == grade) |
                                                  (j == sub_grade) |
                                                  (j == emp_length) |
                                                  (j == home_ownership) |
                                                  (j == purpose)))
    # --------------------------------------------------------------------------------------------------------------------
    fileA = 'modelA.sav'
    fileB = 'modelB.sav'
    fileC = 'modelC.sav'
    modelA = joblib.load(open(fileA, 'rb'))
    modelB = joblib.load(open(fileB, 'rb'))
    modelC = joblib.load(open(fileC, 'rb'))

    X = pd.concat([df_numerik_tes, df_kategorik_tes], axis=1)
    daftar_prediksi = [modelA.predict(X)[0], modelB.predict(X)[
        0], modelC.predict(X)[0]]
    prediksi_final = 'Charged Off' if max(
        daftar_prediksi, key=daftar_prediksi.count) else 'Fully Paid'

    return render_template('prediksi.html', hasil_prediksi=f'{prediksi_final}')


if __name__ == "__main__":
    app.run()
