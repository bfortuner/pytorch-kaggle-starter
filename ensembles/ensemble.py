import os
import pprint
import utils.files
import copy
import constants as c
from predictions.prediction import Prediction



class MegaEnsemblePrediction(Prediction):
    """
    Prediction combining multiple experiments, models and epochs
    """
    def __init__(self, name, pred_type, fpath, thresholds,
                 label_names, val_score, val_probs, val_preds,
                 test_probs, test_preds, created, sub_preds,
                 ens_method, all_val_probs, all_test_probs):
        super().__init__(name, pred_type, fpath, thresholds,
                         label_names, val_score, val_probs, val_preds,
                         test_probs, test_preds, tta=None, created=created,
                         other=None)

        self.sub_preds = self.get_sub_pred_docs(sub_preds)
        self.ens_method = ens_method
        self.all_val_probs = all_val_probs
        self.all_test_probs = all_test_probs

    def get_sub_pred_docs(self, sub_preds):
        docs = []
        for pred in sub_preds:
            docs.append(pred.to_doc(include_exp=False))
        return docs

    def to_doc(self):
        d = copy.deepcopy(self.__dict__)
        d['key'] = self.get_id()
        d['pred_id'] = self.get_id()
        d['display_name'] = self.get_display_name()
        d['preds'] = self.sub_preds
        del d['val_probs']
        del d['val_preds']
        del d['test_probs']
        del d['test_preds']
        del d['all_val_probs']
        del d['all_test_probs']
        return d


