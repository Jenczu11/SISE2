import file_ops as fo
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import json
from keras.callbacks import CSVLogger

from config import NEURAL_NETWORK_SETTINGS, DIRECTORY_FOR_LOG
from normalization import norm, unnorm
from plots import plot_and_save, plot_dist_one_figure, plot_dist_one_axis
from resultshelper import prepare_additional_results_for_excel,save_plots_to_excel
from data_loader import load

def build_model(train_dataset):
    _model = keras.Sequential([
        layers.Dense(NEURAL_NETWORK_SETTINGS.NUMBER_OF_INPUTS, activation='relu',
                     input_shape=[2,]),
        layers.Dense(NEURAL_NETWORK_SETTINGS.NEURONS_FIRST_LAYER, activation='relu'),
        layers.Dense(NEURAL_NETWORK_SETTINGS.NEURONS_SECOND_LAYER, activation='relu'),
        layers.Dense(NEURAL_NETWORK_SETTINGS.NUMBER_OF_OUTPUTS)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    _model.compile(loss='mse',
                   optimizer=optimizer,
                   metrics=['mae', 'mse'])
    return _model

def runApp():
    # Create directory for logs
    fo.createFolder(DIRECTORY_FOR_LOG)
    NEURAL_NETWORK_SETTINGS.save_settings_json(NEURAL_NETWORK_SETTINGS)

    final_test_data, final_test_data_ref, final_test_data_err, final_test_data_measure, final_test_data_to_excel = load()

    file_numbers = [*range(1, 13, 1)]
    train_dataset = fo.concat_xlsx_files_into_data_frame(
    'files/pozyxAPI_dane_pomiarowe/pozyxAPI_only_localization_measurement', file_numbers)
    train_dataset = train_dataset.drop(columns=['0/timestamp', 't', 'no'])
    train_dataset = norm(train_dataset)

    train_labels = train_dataset.drop(columns=['measurement x', 'measurement y'])
    train_dataset = train_dataset.drop(columns=["reference x", "reference y"])

    # Creates instance of csv logger (keras) to monitor model train
    csv_logger = CSVLogger(DIRECTORY_FOR_LOG + "epoch_iteration" + ".csv", separator=';')

    # Create model based on data from @TESTING_DATA_FILENAME
    model = build_model(train_dataset)
    history = model.fit(
        train_dataset, train_labels,
        epochs=NEURAL_NETWORK_SETTINGS.EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[tfdocs.modeling.EpochDots(), csv_logger])
    # Save model to log directory
    model.save(DIRECTORY_FOR_LOG + 'trained_model.h5')

    hist = fo.pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
    print(history)

    # Normalize final data for prediction
    final_test_data_measure = norm(final_test_data_measure)

    # Predict on final data and create dataframe
    final_test_data_predictions = model.predict(final_test_data_measure)
    final_test_data_predictions_dataframe = pd.DataFrame(final_test_data_predictions, columns=['x', 'y'])

    # Unnormalize for plot.
    final_test_data_predictions_dataframe = unnorm(final_test_data_predictions_dataframe)
    final_test_data_measure = unnorm(final_test_data_measure)
    # Plot route prediction
    # Plot predicted
    plot_and_save("prediction", "Predykcja pomiaru", "x", "y", final_test_data_predictions_dataframe)
    # Plot reference
    plot_and_save("reference", "Tor testowy", "reference x", "reference y", final_test_data_ref)
    # Plot gathered data
    plot_and_save("actualdata", "Pomiar", "measurement x", "measurement y", final_test_data_measure)

    # Prepare neural_network_results
    neural_network_results = pd.DataFrame({"x": final_test_data_predictions_dataframe['x'],
                                           "y": final_test_data_predictions_dataframe['y'],
                                           "reference x": final_test_data_ref['reference x'],
                                           "reference y": final_test_data_ref['reference y']
                                           })

    neural_network_results = prepare_additional_results_for_excel(neural_network_results)

    # Plot on two figures
    # sns.relplot(x="błąd", y="% błędnych próbek", kind="line", data=excel)
    # plt.show()
    # sns.relplot(x="błąd", y="% błędnych próbek", kind="line", data=final_test_data_err)
    # plt.show()

    # Plot cumulative distribution function
    plot_dist_one_figure(final_test_data_err, neural_network_results)
    plot_dist_one_axis(final_test_data_err, neural_network_results)

    # Save results to excel
    with pd.ExcelWriter(DIRECTORY_FOR_LOG + 'wyniki.xlsx') as writer:
        final_test_data_to_excel.to_excel(writer, sheet_name='pomiar')
        neural_network_results.to_excel(writer, sheet_name='Wyniki sieci neuronowej')
    save_plots_to_excel()

    # Copy to excel if you want.
    final_test_data_predictions_dataframe.to_clipboard(excel=True)

if __name__ == '__main__':
    for i in range(1):
        runApp()


