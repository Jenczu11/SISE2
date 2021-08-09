import math

import numpy as np
import openpyxl

from config import DIRECTORY_FOR_LOG


def prepare_additional_results_for_excel(df):
    number_of_results = len(df['x'])
    # Create row for 'blad pomiaru'
    df['błąd pomiaru'] = df.apply(
        lambda row: math.sqrt(
            math.pow((row['x'] - row['reference x']), 2) +
            math.pow((row['y'] - row['reference y']), 2)),
        axis=1)
    # Create row for 'blad'
    index = []

    for i in range(number_of_results):
        index.append(i)
    df['błąd'] = index
    # Create row for 'liczba_blednych_probek'
    liczba_blednych_probek = []
    for i in range(number_of_results):
        liczba_blednych_probek.append(np.sum(df['błąd pomiaru'] < i))
    df['liczba błędnych próbek'] = liczba_blednych_probek
    # Create row for '% błędnych próbek'
    df['% błędnych próbek'] = df.apply(lambda row: row['liczba błędnych próbek'] / number_of_results ,axis=1)
    return df

def save_plots_to_excel():
    plotsname = ["dystrybuanta.png","dystrybuanta_sep.png","prediction.png",'reference.png',"actualdata.png"]
    dest_filename = DIRECTORY_FOR_LOG + 'wyniki.xlsx'
    for plot in plotsname:
        print("Saving plot {} to excel".format(plot))
        wb = openpyxl.load_workbook(dest_filename)
        ws = wb.create_sheet(plot)
        ws.title = plot
        img = openpyxl.drawing.image.Image(DIRECTORY_FOR_LOG + plot)
        img.anchor = 'A1' # Or whatever cell location you want to use.
        ws.add_image(img)
        wb.save(dest_filename)
    print("Done saving excel")