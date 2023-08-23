import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import argparse, os, json, sys
import h5py
sys.path.append("../model")
from PFINDataset import PFINDataset
from UQPFIN import UQPFIN as Model

def getprobs(outs):
    alphas = outs + 1
    S = torch.sum(alphas, 1).reshape(-1,1)
    return alphas / S

def index_logic(logic, v1, v2, a):
    if logic == 'AND':
        keep = (a > min(v1, v2)) & (a < max(v1, v2))
    else:
        keep = (a < min(v1, v2)) | (a > max(v1, v2))
    return keep


class ModelEvaluator:
    def __init__(self, model_path, evalMode = True):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dict_path = model_path.replace("_best","").replace("_best","").replace("trained_models/", "trained_model_dicts/") + ".json"
        self.model_dict = json.load(open(self.model_dict_path))
        self.phi_nodes = list(map(int, self.model_dict["phi_nodes"].strip().split(',')))
        self.f_nodes = list(map(int, self.model_dict["f_nodes"].strip().split(',')))
        self.n_phiI = int(self.model_dict['n_phiI'])
        self.label = self.model_dict['label']
        self.data_type = self.model_dict['data_type']
        self.massrange = self.model_dict['massrange']
        self.etarange = self.model_dict['etarange']
        self.ptrange = self.model_dict['ptrange']
        self.x_mode = self.model_dict['x_mode']
        self.skip_labels = self.model_dict["skiplabels"]
        try:
            self.use_softmax = self.model_dict["use_softmax"]
        except:
            self.use_softmax = False
        try:
            self.use_dropout = self.model_dict["use_dropout"]
        except:
            self.use_dropout = False
        self.m_logic, (self.m1, self.m2) = self.massrange.strip().split(':')[0], list(map(float, self.massrange.strip().split(':')[1].split(',')))
        self.pt_logic,(self.pt1, self.pt2) = self.ptrange.strip().split(':')[0], list(map(float, self.ptrange.strip().split(':')[1].split(',')))
        self.eta_logic, (self.eta1, self.eta2) = self.etarange.strip().split(':')[0], list(map(float, self.etarange.strip().split(':')[1].split(',')))
        
        if self.data_type == 'topdata':
            Np = 60
            self.num_classes = 2
            self.skip_labels = []
            self.label_indices = [0,1]
            self.data_path = "../datasets/topdata/test.h5"
        elif self.data_type == 'jetnet':
            Np = 30
            if self.skip_labels:
                self.skip_labels = list(map(int, self.skip_labels.strip().split(',')))
            else:
                self.skip_labels = []
            self.num_classes = 5 - len(self.skip_labels)
            self.label_indices = [i for i in range(5) if i not in self.skip_labels]
            self.data_path = "../datasets/jetnet/test.h5"
        
        
        self.model = Model(particle_feats = 3,
                           n_consts = Np,
                           num_classes = self.num_classes,
                           device = self.device ,
                           PhiI_nodes = self.n_phiI,
                           interaction_mode = self.x_mode,
                           use_softmax = self.use_softmax,
                           use_dropout = self.use_dropout,
                           Phi_sizes = self.phi_nodes,
                           F_sizes   = self.f_nodes).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        if evalMode:
            self.model.eval()
        else:
            self.model.train()
        

        
    def index_groomer(self, a, y):
        keep_masses = index_logic(self.m_logic,   self.m1,   self.m2,   a[:, 1])
        keep_pts    = index_logic(self.pt_logic,  self.pt1,  self.pt2,  a[:, 2])
        keep_etas   = index_logic(self.eta_logic, self.eta1, self.eta2, a[:, 3])

        keep_labels = torch.tensor(np.isin(np.argmax(y.cpu().numpy(), 1), self.skip_labels, invert=True)).bool()

        return (keep_masses & keep_pts & keep_etas & keep_labels).bool()
    
    def evaluate(self, data_loader = None):
        if not data_loader:
            test_set = PFINDataset(self.data_path)
            testloader = DataLoader(test_set, shuffle=True, batch_size=512, num_workers=8, pin_memory=True, persistent_workers=True)
        else:
            testloader = data_loader
        labels = []
        preds = []
        maxprobs = []
        probs = []
        oods = []
        sums = []
        uncs = []
        with torch.no_grad():
            for x,m,a,y in tqdm(testloader, disable = True):
                x = x.cuda()
                m = m.cuda()
                a = a.cuda()
                pred = self.model(x, a, m).cpu()
                idx2keep = self.index_groomer(a.cpu(), y).cpu().numpy()
                if not self.use_softmax:
                    model_prob = getprobs(pred)
                else:
                    model_prob = pred.clone()
                all_prob = torch.zeros_like(y)
                all_prob[:, self.label_indices] = model_prob
                y = y.cpu().numpy()
                all_prob = all_prob.cpu().numpy()
                
                labels.append(np.argmax(y, 1))
                preds.append(np.argmax(all_prob, 1))
                maxprobs.append(all_prob.max(axis=-1))
                probs.append(all_prob)
                oods.append(~idx2keep)
                if not self.use_softmax:
                    sums.append((pred+1).sum(-1).cpu().numpy())
                    
                
        labels = np.concatenate(labels, axis = None)
        preds = np.concatenate(preds, axis = None)
        oods = np.concatenate(oods, axis = None)
        maxprobs = np.concatenate(maxprobs, axis = None)
        probs = np.concatenate(probs, axis = 0)
        if not self.use_softmax:
            sums = np.concatenate(sums, axis = None)
            uncs = len(self.label_indices)*1.0 / sums

        
        if not data_loader:
            del test_set, testloader
        
        return labels, preds, maxprobs, probs, sums, oods, uncs
    

class EnsembleEvaluator:
    def __init__(self, model_paths, data_type = "jetnet"):
        self.model_paths = model_paths
        self.data_type = data_type
        if self.data_type == 'topdata':
            self.data_path = "../datasets/topdata/test.h5"
        elif self.data_type == 'jetnet':
            self.data_path = "../datasets/jetnet/test.h5"
        
    def evaluate(self):
        test_set = PFINDataset(self.data_path)
        testloader = DataLoader(test_set, shuffle=False, batch_size=1000, num_workers=8, pin_memory=True, persistent_workers=True)
        preds = []
        maxprobs = []
        probs = []
        sums = []
        for ii, model_path in enumerate(self.model_paths):
            model_evaluator = ModelEvaluator(model_path, evalMode = True)
            if ii == 0:
                labels, _, _, this_probs, _, oods, _ =  model_evaluator.evaluate(testloader)
            else:
                _, _, _, this_probs, _, _, _ =  model_evaluator.evaluate(testloader)
            probs.append(this_probs[None,:,:])
            del model_evaluator
                
                
        probs = np.concatenate(probs, axis = 0)
        uncs = np.std(probs, axis = 0).max(axis = 1)
        probs = np.mean(probs, axis = 0)
        preds = np.argmax(probs, 1)
        maxprobs = probs.max(axis = 1)
                
        del test_set, testloader
        
        return labels, preds, maxprobs, probs, sums, oods, uncs

 
class MCDOEvaluator:
    def __init__(self, model_path, data_type = "jetnet"):
        self.model_path = model_path
        self.data_type = data_type
        if self.data_type == 'topdata':
            self.data_path = "../datasets/topdata/test.h5"
        elif self.data_type == 'jetnet':
            self.data_path = "../datasets/jetnet/test.h5"
        
    def evaluate(self):
        test_set = PFINDataset(self.data_path)
        testloader = DataLoader(test_set, shuffle=False, batch_size=1000, num_workers=8, pin_memory=True, persistent_workers=True)
        preds = []
        maxprobs = []
        probs = []
        sums = []
        model_evaluator = ModelEvaluator(self.model_path, evalMode = False)
        for ii in range(10):
            if ii == 0:
                labels, _, _, this_probs, _, oods, _ =  model_evaluator.evaluate(testloader)
            else:
                _, _, _, this_probs, _, _, _ =  model_evaluator.evaluate(testloader)
            probs.append(this_probs[None,:,:])
        del model_evaluator
                
                
        probs = np.concatenate(probs, axis = 0)
        uncs = np.std(probs, axis = 0).max(axis = 1)
        probs = np.mean(probs, axis = 0)
        preds = np.argmax(probs, 1)
        maxprobs = probs.max(axis = 1)
                
        del test_set, testloader
        
        return labels, preds, maxprobs, probs, sums, oods, uncs

    
class PlotterTools:
    def __init__(self, model_results, tag):
        self.uncs = model_results['uncs']
        self.preds = model_results['preds']
        self.labels = model_results['labels']
        self.oods = model_results['oods']
        self.tag = tag
        ul, ur = self.uncs.min(), self.uncs.max()
        du = (ur - ul)/100
        self.urange = np.arange(ul+du, ur-du, du)
        
    def ODR_IDAcc(self, ax):
        # plots OOD Detection Rate vs ID Accuracy for differenct unc thresholds
        # ODR = OOD_Positive / Total_OOD
        # IDAcc = Correct_ID_tag / Total_ID
        
        ODRs = []
        IDAccs = []
        
        for u in self.urange:
            odr = (self.oods & (self.uncs >= u)).sum()/self.oods.sum()
            ODRs.append(odr)
            pred_id_indices = (self.uncs < u) & (~self.oods)
            idacc = (self.labels[pred_id_indices] == self.preds[pred_id_indices]).sum()/(~self.oods).sum()
            IDAccs.append(idacc)
            
        ax.plot(IDAccs, ODRs, label=self.tag)
        
    def ODR_Acc(self, ax):
        # plots OOD Detection Rate vs Accuracy for differenct unc thresholds
        # ODR = OOD_Positive / Total_OOD
        # Acc = Correct_OOD or Correctly_tagged_ID / N
        
        ODRs = []
        Accs = []
        
        for u in self.urange:
            odr = (self.oods & (self.uncs >= u)).sum()/self.oods.sum()
            ODRs.append(odr)
            pred_id_indices = self.uncs < u
            acc = ((self.labels[pred_id_indices] == self.preds[pred_id_indices]).sum() + (self.oods & (self.uncs >= u)).sum())/len(self.labels)
            Accs.append(acc)
            
        ax.plot(Accs, ODRs, label=self.tag)
        
    def OCR_ODR(self, ax):
        # plots OOD Detection Rate vs OOD Conf Rate for differenct unc thresholds
        # ODR = OOD_Positive / Total_OOD (like recall or True-Positive-Rate)
        # OCR = OOD_Positive / Total_Pred_OOD (like Precision)
        
        ODRs = []
        OCRs = []
        
        for u in self.urange:
            odr = (self.oods & (self.uncs >= u)).sum()/self.oods.sum()
            ODRs.append(odr)
            ocr = (self.oods & (self.uncs >= u)).sum()/(self.uncs >= u).sum()
            OCRs.append(ocr)
            
        ax.plot(ODRs, OCRs, label=self.tag)
        
    def ODR_IMR(self, ax):
        # plots OOD Detection Rate vs ID Mistag Rate for differenct unc thresholds (equivalent to ROC)
        # ODR = OOD_Positive / Total_OOD
        # IMR = ID_Positive / Total_ID
        
        ODRs = []
        IMRs = []
        
        for u in self.urange:
            odr = (self.oods & (self.uncs >= u)).sum()/self.oods.sum()
            ODRs.append(odr)
            pred_id_indices = (self.uncs >= u) & (~self.oods)
            imr = pred_id_indices.sum()/(~self.oods).sum()
            IMRs.append(imr)
            
        ax.plot(IMRs, ODRs, label=self.tag)
        
        
    def RAR_RER(self, ax):
        #plots Remaining Accuracy Rate v Remaining Error Rate [https://ceur-ws.org/Vol-2640/paper_18.pdf]
        # RAR = Correct and Confident / N = [(large unc and OOD) or (small uncertainty and correct ID)] / N
        # RER = Incorrect and Confident / N = [(large unc and ID) or (small unc and OOD) ] / N
        
        RARs = []
        RERs = []
        
        for u in self.urange:
            CC_ood = ((self.uncs >= u) & self.oods).sum()
            CC_id_indices = (self.uncs < u) & (~self.oods)
            CC_id = (self.labels[CC_id_indices] == self.preds[CC_id_indices]).sum()
            rar = (CC_ood + CC_id)/len(self.labels)
            
            IC_id = ((self.uncs >= u) & ~self.oods).sum()
            IC_ood = ((self.uncs < u) & self.oods).sum()
            rer = (IC_ood + IC_id)/len(self.labels)
            
            RARs.append(rar)
            RERs.append(rer)
            
        ax.plot(RERs, RARs, label=self.tag)
        
        
        
        
        
        