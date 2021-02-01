import numpy as np
import json

import matplotlib.pyplot as plt


class MLEstimator:
    def __init__(self, top_k):
        self.top_k = top_k
        self.model_perf_list = None
        self.acc_list = list()
        self.epoch_list = list()
        self.weight_list = list()

    def import_accuracy_dataset(self, dataset_path):
        with open(dataset_path) as json_file:
            self.model_perf_list = json.load(json_file)

    def predict_accuracy(self, input_model_dict, input_model_epoch):
        neighbour_model_list = self.compute_model_similarity(input_model_dict, self.model_perf_list)

        for model in neighbour_model_list:
            for aidx, acc in enumerate(model['accuracy']):
                self.acc_list.append(acc)
                self.epoch_list.append(aidx)

        self.weight_list = [1 / len(self.acc_list)] * len(self.acc_list)

        coefs = np.polyfit(x=np.asarray(self.epoch_list), y=np.asarray(self.acc_list), deg=3, w=self.weight_list)

        '''
        print(np.asarray(self.acc_list))

        plt.figure()
        plt.plot(np.arange(1, 21), np.polyval(coefs, np.arange(1, 21)), color="black")
        plt.show()
        
        '''
        acc_estimation = np.polyval(coefs, input_model_epoch)

        return acc_estimation

    def compute_model_similarity(self, center_model, candidate_models):
        ''' similarity between each model in mlbase and the center model '''
        similarity_list = list()

        for candidate in candidate_models:
            ''' 
                We only take the candidate model that has: 
                1. same training dataset 
                2. same output class 
                3. same learning rate
                4. same optimizer 
                Otherwise, set the similarity as -1
            '''
            if (center_model['training_data'] == candidate['training_data'] and
                center_model['classes'] == candidate['classes'] and
                center_model['learn_rate'] == candidate['classes'] and
                center_model['opt'] == candidate['opt']):

                max_x = max([center_model['num_parameters'], candidate['num_parameters']])
                diff_x = np.abs(center_model['num_parameters'] - candidate['num_parameters'])
                parameter_similarity = 1 - diff_x / max_x
                similarity_list.append(parameter_similarity)
            else:
                similarity_list.append(-1)

        ''' get the top k models from mlbase according to the similarity '''
        similarity_sorted_idx_list = sorted(range(len(similarity_list)),
                                            key=lambda k: similarity_list[k],
                                            reverse=True)[:self.top_k]

        topk_model_list = [self.model_perf_list[i] for i in similarity_sorted_idx_list]

        return topk_model_list


if __name__ == "__main__":
    input_model = dict()

    input_model['model_name'] = 'model_demo'
    input_model['num_parameters'] = 3000000
    input_model['batch_size'] = 32
    input_model['opt'] = 'SGD'
    input_model['learn_rate'] = 0.01
    input_model['training_data'] = 'cifar'
    input_model['classes'] = 10

    ml_estimator = MLEstimator(top_k=10)
    ml_estimator.import_accuracy_dataset('/home/ruiliu/Development/ml-estimator/mlbase/model_acc.json')
    acc = ml_estimator.predict_accuracy(input_model, 10)

    print(acc)
