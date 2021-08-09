import datetime
import json

class NEURAL_NETWORK_SETTINGS:
    NUMBER_OF_INPUTS=6
    NEURONS_FIRST_LAYER=6
    NEURONS_SECOND_LAYER=6
    NUMBER_OF_OUTPUTS=2
    EPOCHS = 100
    NORMALIZATION=True
    # Maximum abs value for data
    # NORMALIZATION_DIVIDER=6964
    # All files
    NORMALIZATION_DIVIDER=7081
    TESTING_DATA_FILENAME = 'files/pozyxAPI_dane_pomiarowe/pozyxAPI_only_localization_measurement1.xlsx'
    def save_settings_json(self):
        data = {}
        data['settings'] = []
        data['settings'].append({
            'NUMBER_OF_INPUTS': self.NUMBER_OF_INPUTS,
            'NEURONS_FIRST_LAYER': self.NEURONS_FIRST_LAYER,
            'NEURONS_SECOND_LAYER': self.NEURONS_SECOND_LAYER,
            'NUMBER_OF_OUTPUTS': self.NUMBER_OF_OUTPUTS,
            'EPOCHS': self.EPOCHS,
            'NORMALIZATION': self.NORMALIZATION,
            'NORMALIZATION_DIVIDER': self.NORMALIZATION_DIVIDER,
            'TESTING_DATA_FILENAME': self.TESTING_DATA_FILENAME,
        })
        with open(DIRECTORY_FOR_LOG+'settings.json', 'w') as outfile:
            json.dump(data, outfile)

DIRECTORY_FOR_LOG = "files/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

# def save_settings_json():
#     data = {}
#     data['settings'] = []
#     data['settings'].append({
#         'NUMBER_OF_INPUTS': NUMBER_OF_INPUTS,
#         'NEURONS_FIRST_LAYER': NEURONS_FIRST_LAYER,
#         'NEURONS_SECOND_LAYER': NEURONS_SECOND_LAYER,
#         'NUMBER_OF_OUTPUTS': NUMBER_OF_OUTPUTS,
#         'NORMALIZATION_DIVIDER': NORMALIZATION_DIVIDER,
#     })
#
#     with open('data.txt', 'w') as outfile:
#         json.dump(data, outfile)
