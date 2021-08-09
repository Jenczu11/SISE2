import file_ops as fo
def load():
    final_test_data = fo.xlsx_to_dataset('files/pozyxAPI_only_localization_dane_testowe_i_dystrybuanta.xlsx')

    final_test_data_ref = final_test_data.drop(columns=['0/timestamp', 't', 'no', 'measurement x', 'measurement y',
                                                        'błąd pomiaru', 'liczba błędnych próbek', '% błędnych próbek',
                                                        'błąd', 'Unnamed: 9'])

    final_test_data_err = final_test_data.drop(columns=['0/timestamp', 't', 'no', 'measurement x', 'measurement y',
                                                        'błąd pomiaru', 'liczba błędnych próbek', 'reference x',
                                                        'reference y', 'Unnamed: 9'])

    final_test_data_measure = final_test_data.drop(columns=['0/timestamp', 't', 'no', 'reference x', 'reference y',
                                                            'błąd pomiaru', 'liczba błędnych próbek',
                                                            '% błędnych próbek',
                                                            'błąd', 'Unnamed: 9'])

    final_test_data_to_excel = final_test_data.drop(columns=['0/timestamp', 't', 'no', 'Unnamed: 9']);

    # Drop any NaN in dataframe
    final_test_data = final_test_data.dropna()
    final_test_data_ref = final_test_data_ref.dropna()
    final_test_data_err = final_test_data_err.dropna()
    final_test_data_measure = final_test_data_measure.dropna()
    final_test_data_to_excel = final_test_data_to_excel.dropna()

    return final_test_data,final_test_data_ref,final_test_data_err,final_test_data_measure,final_test_data_to_excel