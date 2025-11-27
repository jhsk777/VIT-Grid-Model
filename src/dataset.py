import numpy as np
import torch
import xarray as xr
from datetime import datetime, timedelta
import os

# assign PM2.5 class accorrding to the range
def assign_class(arr, range, classes):
  return np.select([np.logical_and(arr > r[0], arr <= r[1]) for r in range], classes, default=-1)

def assign_class2(pm_arr, pm_mask, range, classes):
    cur_class = np.select([np.logical_and(pm_arr > r[0], pm_arr <= r[1]) for r in range], classes, default=-1)
    cur_class[~pm_mask] = -1
    return cur_class

class Air_with_fixed_Sat_Dataset(torch.utils.data.Dataset):
    """
    :param times: list of time in the dataset
    :param time_index: index of time in the dataset
    :param sat_outputs: satellite predictions
    :param sat_inputs: satellite observations
    :param feats,masks: features and corresponding masks
    :param input_dim: length of time series input
    :param output_dim: length of time series output
    :param prev_len: length of previous values for normalization
    :param korea_stn_num,china_stn_num: number of stations in Korea and China
    """
    def __init__(self, times, sat_outputs, sat_inputs, feats, masks, input_dim, output_dim, prev_len, korea_stn_num, china_stn_num):
        
        self.times = times
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prev_len = prev_len
        self.korea_stn_num = korea_stn_num
        self.china_stn_num = china_stn_num
        self.total_stn_num = korea_stn_num + china_stn_num
        
        self.feat_torch = torch.FloatTensor(feats)
        self.mask_torch = torch.from_numpy(masks)
        self.sat_outputs_torch = torch.FloatTensor(sat_outputs)
        self.sat_inputs_torch = torch.FloatTensor(sat_inputs)

    
    def load_feats(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        feats = self.feat_torch[mod_idx - self.input_dim + 1: mod_idx + 1]
        return feats
    
    def load_masks(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        masks = self.mask_torch[mod_idx - self.input_dim + 1: mod_idx + self.output_dim + 1].type(torch.BoolTensor)
        return masks
    
    def __len__(self) -> int:
        return (len(self.times) - (self.prev_len - 1) - self.output_dim)
    
    def __getitem__(self, idx):
        mod_idx = idx + (self.prev_len - 1)

        feats = self.load_feats(idx)
        masks = self.load_masks(idx)

        sat_outputs = self.sat_outputs_torch[mod_idx]
        sat_inputs = self.sat_inputs_torch[mod_idx]

        pred_pm25_vals = self.feat_torch[mod_idx+1:mod_idx+1+self.output_dim, :self.korea_stn_num, 0].numpy()
        pred_pm25_mask = self.feat_torch[mod_idx+1:mod_idx+1+self.output_dim, :self.korea_stn_num, 6].numpy().astype(bool)

        pred_pm25_mask = ~pred_pm25_mask

        prev_pm25_vals = self.feat_torch[mod_idx-self.prev_len + 1:mod_idx + 1, :, 0].numpy()

        raw_times = []

        for t_idx in range(self.input_dim + self.output_dim):
            cur_idx = mod_idx - self.input_dim + 1 + t_idx
            raw_times.append([self.times[cur_idx].year, self.times[cur_idx].month, self.times[cur_idx].day, self.times[cur_idx].hour])

        range_4class = [(-1,15),(15,35),(35,75), (75,np.Inf)]
        class_four = [0,1,2,3]
        pred_pm25_class = assign_class2(pred_pm25_vals, pred_pm25_mask, range_4class, class_four)
        
        return feats, masks, sat_outputs, sat_inputs, torch.IntTensor(pred_pm25_class), torch.FloatTensor(pred_pm25_vals), torch.BoolTensor(pred_pm25_mask), torch.FloatTensor(raw_times), torch.FloatTensor(prev_pm25_vals)
    
    def collate_fn(self, samples):
        feats, masks, sat_outputs, sat_inputs, pred_classes, pred_vals, pred_mask, raw_times, prev_vals = zip(*samples)
        feats = torch.stack(feats, dim=0)
        masks = torch.stack(masks, dim=0)
        sat_outputs = torch.stack(sat_outputs, dim=0)
        sat_inputs = torch.stack(sat_inputs, dim=0)
        pred_classes = torch.stack(pred_classes, dim=0)
        pred_vals = torch.stack(pred_vals, dim=0)
        pred_mask = torch.stack(pred_mask, dim=0)
        raw_times = torch.stack(raw_times)
        prev_vals = torch.stack(prev_vals, dim=0)
        return feats, masks, sat_outputs, sat_inputs, pred_classes, pred_vals, pred_mask, raw_times, prev_vals

class Air_with_Simulation_Dataset(torch.utils.data.Dataset):
    """
    :param times: list of time in the dataset
    :param time_index: index of time in the dataset
    :param sat_outputs: satellite predictions
    :param sat_inputs: satellite observations
    :param feats,masks: features and corresponding masks
    :param input_dim: length of time series input
    :param output_dim: length of time series output
    :param prev_len: length of previous values for normalization
    :param korea_stn_num,china_stn_num: number of stations in Korea and China
    """
    def __init__(self, times, feats, masks, simulation, input_dim, output_dim, prev_len, korea_stn_num, china_stn_num):
        
        self.times = times
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prev_len = prev_len
        self.korea_stn_num = korea_stn_num
        self.china_stn_num = china_stn_num
        self.total_stn_num = korea_stn_num + china_stn_num
        
        self.feat_torch = torch.FloatTensor(feats)
        self.mask_torch = torch.from_numpy(masks)
        self.simluation_torch = torch.FloatTensor(simulation)

    
    def load_feats(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        feats = self.feat_torch[mod_idx - self.input_dim + 1: mod_idx + 1]
        return feats
    
    def load_masks(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        masks = self.mask_torch[mod_idx - self.input_dim + 1: mod_idx + self.output_dim + 1].type(torch.BoolTensor)
        return masks
    
    def __len__(self) -> int:
        return (len(self.times) - (self.prev_len - 1) - self.output_dim)
    
    def __getitem__(self, idx):

        mod_idx = idx + (self.prev_len - 1)

        feats = self.load_feats(idx)
        masks = self.load_masks(idx)
        simulation = self.simluation_torch[mod_idx]

        pred_pm25_vals = self.feat_torch[mod_idx+1:mod_idx+1+self.output_dim, :self.korea_stn_num, 0].numpy()
        pred_pm25_mask = self.feat_torch[mod_idx+1:mod_idx+1+self.output_dim, :self.korea_stn_num, 6].numpy().astype(bool)

        pred_pm25_mask = ~pred_pm25_mask

        prev_pm25_vals = self.feat_torch[mod_idx-self.prev_len + 1:mod_idx + 1, :, 0].numpy()

        raw_times = []

        for t_idx in range(self.input_dim + self.output_dim):
            cur_idx = mod_idx - self.input_dim + 1 + t_idx
            raw_times.append([self.times[cur_idx].year, self.times[cur_idx].month, self.times[cur_idx].day, self.times[cur_idx].hour])

        range_4class = [(-1,15),(15,35),(35,75), (75,np.Inf)]
        class_four = [0,1,2,3]
        pred_pm25_class = assign_class2(pred_pm25_vals, pred_pm25_mask, range_4class, class_four)
        
        return feats, masks, simulation, torch.IntTensor(pred_pm25_class), torch.FloatTensor(pred_pm25_vals), torch.BoolTensor(pred_pm25_mask), torch.FloatTensor(raw_times), torch.FloatTensor(prev_pm25_vals)
    
    def collate_fn(self, samples):
        feats, masks, simulation, pred_classes, pred_vals, pred_mask, raw_times, prev_vals = zip(*samples)
        feats = torch.stack(feats, dim=0)
        masks = torch.stack(masks, dim=0)
        simulation = torch.stack(simulation, dim=0)
        pred_classes = torch.stack(pred_classes, dim=0)
        pred_vals = torch.stack(pred_vals, dim=0)
        pred_mask = torch.stack(pred_mask, dim=0)
        raw_times = torch.stack(raw_times)
        prev_vals = torch.stack(prev_vals, dim=0)
        return feats, masks, simulation, pred_classes, pred_vals, pred_mask, raw_times, prev_vals

class Air_only_Dataset(torch.utils.data.Dataset):
    """
    :param times: list of time in the dataset
    :param time_index: index of time in the dataset
    :param sat_outputs: satellite predictions
    :param sat_inputs: satellite observations
    :param feats,masks: features and corresponding masks
    :param input_dim: length of time series input
    :param output_dim: length of time series output
    :param prev_len: length of previous values for normalization
    :param korea_stn_num,china_stn_num: number of stations in Korea and China
    """
    def __init__(self, times, feats, masks, input_dim, output_dim, prev_len, korea_stn_num, china_stn_num):
        
        self.times = times
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prev_len = prev_len
        self.korea_stn_num = korea_stn_num
        self.china_stn_num = china_stn_num
        self.total_stn_num = korea_stn_num + china_stn_num
        
        self.feat_torch = torch.FloatTensor(feats)
        self.mask_torch = torch.from_numpy(masks)

    
    def load_feats(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        feats = self.feat_torch[mod_idx - self.input_dim + 1: mod_idx + 1]
        return feats
    
    def load_masks(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        masks = self.mask_torch[mod_idx - self.input_dim + 1: mod_idx + self.output_dim + 1].type(torch.BoolTensor)
        return masks
    
    def __len__(self) -> int:
        return (len(self.times) - (self.prev_len - 1) - self.output_dim)
    
    def __getitem__(self, idx):

        mod_idx = idx + (self.prev_len - 1)

        feats = self.load_feats(idx)
        masks = self.load_masks(idx)

        pred_pm25_vals = self.feat_torch[mod_idx+1:mod_idx+1+self.output_dim, :self.korea_stn_num, 0].numpy()
        pred_pm25_mask = self.feat_torch[mod_idx+1:mod_idx+1+self.output_dim, :self.korea_stn_num, 6].numpy().astype(bool)

        pred_pm25_mask = ~pred_pm25_mask

        prev_pm25_vals = self.feat_torch[mod_idx-self.prev_len + 1:mod_idx + 1, :, 0].numpy()

        raw_times = []

        for t_idx in range(self.input_dim + self.output_dim):
            cur_idx = mod_idx - self.input_dim + 1 + t_idx
            raw_times.append([self.times[cur_idx].year, self.times[cur_idx].month, self.times[cur_idx].day, self.times[cur_idx].hour])

        range_4class = [(-1,15),(15,35),(35,75), (75,np.Inf)]
        class_four = [0,1,2,3]
        pred_pm25_class = assign_class2(pred_pm25_vals, pred_pm25_mask, range_4class, class_four)
        
        return feats, masks, torch.IntTensor(pred_pm25_class), torch.FloatTensor(pred_pm25_vals), torch.BoolTensor(pred_pm25_mask), torch.FloatTensor(raw_times), torch.FloatTensor(prev_pm25_vals)
    
    def collate_fn(self, samples):
        feats, masks, pred_classes, pred_vals, pred_mask, raw_times, prev_vals = zip(*samples)
        feats = torch.stack(feats, dim=0)
        masks = torch.stack(masks, dim=0)
        pred_classes = torch.stack(pred_classes, dim=0)
        pred_vals = torch.stack(pred_vals, dim=0)
        pred_mask = torch.stack(pred_mask, dim=0)
        raw_times = torch.stack(raw_times)
        prev_vals = torch.stack(prev_vals, dim=0)
        return feats, masks, pred_classes, pred_vals, pred_mask, raw_times, prev_vals

class Air_with_Simulation_Dataset_v2(torch.utils.data.Dataset):
    """
    :param times: list of time in the dataset
    :param time_index: index of time in the dataset
    :param sat_outputs: satellite predictions
    :param sat_inputs: satellite observations
    :param feats,masks: features and corresponding masks
    :param input_dim: length of time series input
    :param output_dim: length of time series output
    :param prev_len: length of previous values for normalization
    :param korea_stn_num,china_stn_num: number of stations in Korea and China
    """
    def __init__(self, times, feats, masks, simulation, simulation_pm, input_dim, output_dim, prev_len, korea_stn_num, china_stn_num):
        
        self.times = times
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prev_len = prev_len
        self.korea_stn_num = korea_stn_num
        self.china_stn_num = china_stn_num
        self.total_stn_num = korea_stn_num + china_stn_num
        
        self.feat_torch = torch.FloatTensor(feats)
        self.mask_torch = torch.from_numpy(masks)
        self.simluation_torch = torch.FloatTensor(simulation)
        self.simluation_pm_torch = torch.FloatTensor(simulation_pm)

    
    def load_feats(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        feats = self.feat_torch[mod_idx - self.input_dim + 1: mod_idx + 1]
        return feats
    
    def load_masks(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        masks = self.mask_torch[mod_idx - self.input_dim + 1: mod_idx + self.output_dim + 1].type(torch.BoolTensor)
        return masks
    
    def __len__(self) -> int:
        return (len(self.times) - (self.prev_len - 1) - self.output_dim)
    
    def __getitem__(self, idx):

        mod_idx = idx + (self.prev_len - 1)

        feats = self.load_feats(idx)
        masks = self.load_masks(idx)
        simulation = self.simluation_torch[mod_idx]
        simulation_pm = self.simluation_pm_torch[mod_idx]

        pred_pm25_vals = self.feat_torch[mod_idx+1:mod_idx+1+self.output_dim, :self.korea_stn_num, 0].numpy()
        pred_pm25_mask = self.feat_torch[mod_idx+1:mod_idx+1+self.output_dim, :self.korea_stn_num, 6].numpy().astype(bool)

        pred_pm25_mask = ~pred_pm25_mask

        prev_pm25_vals = self.feat_torch[mod_idx-self.prev_len + 1:mod_idx + 1, :, 0].numpy()

        raw_times = []

        for t_idx in range(self.input_dim + self.output_dim):
            cur_idx = mod_idx - self.input_dim + 1 + t_idx
            raw_times.append([self.times[cur_idx].year, self.times[cur_idx].month, self.times[cur_idx].day, self.times[cur_idx].hour])

        range_4class = [(-1,15),(15,35),(35,75), (75,np.Inf)]
        class_four = [0,1,2,3]
        pred_pm25_class = assign_class2(pred_pm25_vals, pred_pm25_mask, range_4class, class_four)
        
        return feats, masks, simulation, simulation_pm, torch.IntTensor(pred_pm25_class), torch.FloatTensor(pred_pm25_vals), torch.BoolTensor(pred_pm25_mask), torch.FloatTensor(raw_times), torch.FloatTensor(prev_pm25_vals)
    
    def collate_fn(self, samples):
        feats, masks, simulation, simulation_pm, pred_classes, pred_vals, pred_mask, raw_times, prev_vals = zip(*samples)
        feats = torch.stack(feats, dim=0)
        masks = torch.stack(masks, dim=0)
        simulation = torch.stack(simulation, dim=0)
        simulation_pm = torch.stack(simulation_pm, dim=0)
        pred_classes = torch.stack(pred_classes, dim=0)
        pred_vals = torch.stack(pred_vals, dim=0)
        pred_mask = torch.stack(pred_mask, dim=0)
        raw_times = torch.stack(raw_times)
        prev_vals = torch.stack(prev_vals, dim=0)
        return feats, masks, simulation, simulation_pm, pred_classes, pred_vals, pred_mask, raw_times, prev_vals
    
class Air_Simulation_Reanalysis_Dataset(torch.utils.data.Dataset):
    """
    :param times: list of time in the dataset
    :param time_index: index of time in the dataset
    :param sat_outputs: satellite predictions
    :param sat_inputs: satellite observations
    :param feats,masks: features and corresponding masks
    :param input_dim: length of time series input
    :param output_dim: length of time series output
    :param prev_len: length of previous values for normalization
    :param korea_stn_num,china_stn_num: number of stations in Korea and China
    """
    def __init__(self, times, feats, masks, simulation, reanlysis, input_dim, output_dim, prev_len, korea_stn_num, china_stn_num):
        
        self.times = times
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prev_len = prev_len
        self.korea_stn_num = korea_stn_num
        self.china_stn_num = china_stn_num
        self.total_stn_num = korea_stn_num + china_stn_num
        
        self.feat_torch = torch.FloatTensor(feats)
        self.mask_torch = torch.from_numpy(masks)
        self.simluation_torch = torch.FloatTensor(simulation)
        self.reanlysis_torch = torch.FloatTensor(reanlysis)

    
    def load_feats(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        feats = self.feat_torch[mod_idx - self.input_dim + 1: mod_idx + 1]
        return feats
    
    def load_masks(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        masks = self.mask_torch[mod_idx - self.input_dim + 1: mod_idx + self.output_dim + 1].type(torch.BoolTensor)
        return masks
    
    def __len__(self) -> int:
        return (len(self.times) - (self.prev_len - 1) - self.output_dim)
    
    def __getitem__(self, idx):

        mod_idx = idx + (self.prev_len - 1)

        feats = self.load_feats(idx)
        masks = self.load_masks(idx)
        simulation = self.simluation_torch[mod_idx]
        reanalysis = self.reanlysis_torch[mod_idx+1:mod_idx+1+self.output_dim]

        prev_pm25_vals = self.feat_torch[mod_idx-self.prev_len + 1:mod_idx + 1, :, 0].numpy()

        raw_times = []

        for t_idx in range(self.input_dim + self.output_dim):
            cur_idx = mod_idx - self.input_dim + 1 + t_idx
            raw_times.append([self.times[cur_idx].year, self.times[cur_idx].month, self.times[cur_idx].day, self.times[cur_idx].hour])

        range_4class = [(-1,15),(15,35),(35,75), (75,np.Inf)]
        class_four = [0,1,2,3]
        reanalysis_class = assign_class(reanalysis.numpy(), range_4class, class_four)

        return feats, masks, simulation, reanalysis, torch.IntTensor(reanalysis_class), torch.FloatTensor(raw_times), torch.FloatTensor(prev_pm25_vals)

    def collate_fn(self, samples):
        feats, masks, simulation, reanalysis, reanalysis_class, raw_times, prev_vals = zip(*samples)
        feats = torch.stack(feats, dim=0)
        masks = torch.stack(masks, dim=0)
        simulation = torch.stack(simulation, dim=0)
        reanalysis = torch.stack(reanalysis, dim=0)
        reanalysis_class = torch.stack(reanalysis_class, dim=0)
        raw_times = torch.stack(raw_times)
        prev_vals = torch.stack(prev_vals, dim=0)
        return feats, masks, simulation, reanalysis, reanalysis_class, raw_times, prev_vals

class Air_Simulation_Reanalysis_Dataset_w_curr(torch.utils.data.Dataset):
    """
    :param times: list of time in the dataset
    :param time_index: index of time in the dataset
    :param sat_outputs: satellite predictions
    :param sat_inputs: satellite observations
    :param feats,masks: features and corresponding masks
    :param input_dim: length of time series input
    :param output_dim: length of time series output
    :param prev_len: length of previous values for normalization
    :param korea_stn_num,china_stn_num: number of stations in Korea and China
    """
    def __init__(self, times, feats, masks, simulation, reanlysis, input_dim, output_dim, prev_len, korea_stn_num, china_stn_num):
        
        self.times = times
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prev_len = prev_len
        self.korea_stn_num = korea_stn_num
        self.china_stn_num = china_stn_num
        self.total_stn_num = korea_stn_num + china_stn_num
        
        self.feat_torch = torch.FloatTensor(feats)
        self.mask_torch = torch.from_numpy(masks)
        self.simluation_torch = torch.FloatTensor(simulation)
        self.reanlysis_torch = torch.FloatTensor(reanlysis)

    
    def load_feats(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        feats = self.feat_torch[mod_idx - self.input_dim + 1: mod_idx + 1]
        return feats
    
    def load_masks(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        masks = self.mask_torch[mod_idx - self.input_dim + 1: mod_idx + self.output_dim + 1].type(torch.BoolTensor)
        return masks
    
    def __len__(self) -> int:
        return (len(self.times) - (self.prev_len - 1) - self.output_dim)
    
    def __getitem__(self, idx):

        mod_idx = idx + (self.prev_len - 1)

        feats = self.load_feats(idx)
        masks = self.load_masks(idx)
        simulation = self.simluation_torch[mod_idx]
        reanalysis = self.reanlysis_torch[mod_idx+1:mod_idx+1+self.output_dim]

        curr_reanalysis = self.reanlysis_torch[mod_idx]

        prev_pm25_vals = self.feat_torch[mod_idx-self.prev_len + 1:mod_idx + 1, :, 0].numpy()

        raw_times = []

        for t_idx in range(self.input_dim + self.output_dim):
            cur_idx = mod_idx - self.input_dim + 1 + t_idx
            raw_times.append([self.times[cur_idx].year, self.times[cur_idx].month, self.times[cur_idx].day, self.times[cur_idx].hour])

        range_4class = [(-1,15),(15,35),(35,75), (75,np.Inf)]
        class_four = [0,1,2,3]
        reanalysis_class = assign_class(reanalysis.numpy(), range_4class, class_four)

        return feats, masks, simulation, curr_reanalysis, reanalysis, torch.IntTensor(reanalysis_class), torch.FloatTensor(raw_times), torch.FloatTensor(prev_pm25_vals)

    def collate_fn(self, samples):
        feats, masks, simulation, curr_reanalysis, reanalysis, reanalysis_class, raw_times, prev_vals = zip(*samples)
        feats = torch.stack(feats, dim=0)
        masks = torch.stack(masks, dim=0)
        simulation = torch.stack(simulation, dim=0)
        curr_reanalysis = torch.stack(curr_reanalysis, dim=0)
        reanalysis = torch.stack(reanalysis, dim=0)
        reanalysis_class = torch.stack(reanalysis_class, dim=0)
        raw_times = torch.stack(raw_times)
        prev_vals = torch.stack(prev_vals, dim=0)
        return feats, masks, simulation, curr_reanalysis, reanalysis, reanalysis_class, raw_times, prev_vals

class Air_Simulation_Reanalysis_Dataset_v2(torch.utils.data.Dataset):
    """
    :param times: list of time in the dataset
    :param time_index: index of time in the dataset
    :param sat_outputs: satellite predictions
    :param sat_inputs: satellite observations
    :param feats,masks: features and corresponding masks
    :param input_dim: length of time series input
    :param output_dim: length of time series output
    :param prev_len: length of previous values for normalization
    :param korea_stn_num,china_stn_num: number of stations in Korea and China
    """
    def __init__(self, times, feats, masks, input_dim, output_dim, prev_len, korea_stn_num, china_stn_num, cmaq_size, sim_data_path, reanalysis_data_path, feat_infos):
        
        self.times = times
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prev_len = prev_len
        self.korea_stn_num = korea_stn_num
        self.china_stn_num = china_stn_num
        self.total_stn_num = korea_stn_num + china_stn_num
        self.cmaq_size = cmaq_size
        self.sim_data_path = sim_data_path
        self.reanalysis_data_path = reanalysis_data_path
        self.feat_infos = feat_infos
        
        self.feat_torch = torch.FloatTensor(feats)
        self.mask_torch = torch.from_numpy(masks)

    
    def load_feats(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        feats = self.feat_torch[mod_idx - self.input_dim + 1: mod_idx + 1]
        return feats
    
    def load_masks(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        masks = self.mask_torch[mod_idx - self.input_dim + 1: mod_idx + self.output_dim + 1].type(torch.BoolTensor)
        return masks
    
    def __len__(self) -> int:
        return (len(self.times) - (self.prev_len - 1) - self.output_dim)
    
    def __getitem__(self, idx):

        mod_idx = idx + (self.prev_len - 1)

        feats = self.load_feats(idx)
        masks = self.load_masks(idx)

        prev_pm25_vals = self.feat_torch[mod_idx-self.prev_len + 1:mod_idx + 1, :, 0].numpy()

        raw_times = []

        feat_dim = self.feat_torch.shape[-1]

        for t_idx in range(self.input_dim + self.output_dim):
            cur_idx = mod_idx - self.input_dim + 1 + t_idx
            raw_times.append([self.times[cur_idx].year, self.times[cur_idx].month, self.times[cur_idx].day, self.times[cur_idx].hour])

        simulation = torch.zeros(self.cmaq_size[0], self.cmaq_size[1], self.output_dim * ((feat_dim // 2) * 4 + 4))
        reanalysis = torch.zeros(self.output_dim, self.cmaq_size[0], self.cmaq_size[1])

        for t_idx in range(self.output_dim):
            
            cur_t_utc = self.times[mod_idx] - timedelta(hours=9) + timedelta(hours=t_idx + 1)
            cur_f_name = f"{self.reanalysis_data_path}/{cur_t_utc.year}/ACONC.PM_RQ40i8a.KNU_09_01.{cur_t_utc.strftime('%Y%m%d')}.nc"
            ds = xr.open_dataset(cur_f_name)
            cur_pm25 = ds['PM2P5'].values
            cur_pm25 = cur_pm25[cur_t_utc.hour, 0]
            reanalysis[t_idx] = torch.FloatTensor(cur_pm25)

            cur_idx = mod_idx + t_idx + 1
            cur_t_utc = self.times[cur_idx] - timedelta(hours=9)
            lead_from_03h = cur_t_utc.hour + 21
            if cur_t_utc.hour >= 3:
                date_03h = cur_t_utc - timedelta(days=1)
            else:
                date_03h = cur_t_utc - timedelta(days=2)
                lead_from_03h += 24
            lead_from_09h = cur_t_utc.hour + 15
            if lead_from_09h >= 18:
                date_09h = cur_t_utc - timedelta(days=1)
            else:
                date_09h = cur_t_utc - timedelta(days=2)
                lead_from_09h += 24
            lead_from_15h = cur_t_utc.hour + 9
            if lead_from_15h >= 12:
                date_15h = cur_t_utc - timedelta(days=1)
            else:
                date_15h = cur_t_utc - timedelta(days=2)
                lead_from_15h += 24
            lead_from_21h = cur_t_utc.hour + 3
            if lead_from_21h >= 6:
                date_21h = cur_t_utc - timedelta(days=1)
            else:
                date_21h = cur_t_utc - timedelta(days=2)
                lead_from_21h += 24
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4] = lead_from_03h
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 1] = lead_from_09h
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 2] = lead_from_15h
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 3] = lead_from_21h

            cur_sim_f_03h_name = f"{self.sim_data_path}/{date_03h.year}/" + date_03h.strftime("%m%d") + f"03_{lead_from_03h:02d}.npy"
            if not os.path.exists(cur_sim_f_03h_name):
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_03h = np.load(cur_sim_f_03h_name)
            if len(cur_sim_data_03h.shape) != 3:
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))

            cur_sim_data_03h[0] = (cur_sim_data_03h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_03h[1] = (cur_sim_data_03h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_03h[2] = (cur_sim_data_03h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_03h[3] = (cur_sim_data_03h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_03h[5] = (cur_sim_data_03h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]

            cur_sim_data_03h = np.moveaxis(cur_sim_data_03h, 0, -1)
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4): t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2)] = torch.FloatTensor(cur_sim_data_03h)

            cur_sim_f_09h_name = f"{self.sim_data_path}/{date_09h.year}/" + date_09h.strftime("%m%d") + f"09_{lead_from_09h:02d}.npy"
            if not os.path.exists(cur_sim_f_09h_name):
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_09h = np.load(cur_sim_f_09h_name)
            if len(cur_sim_data_09h.shape) != 3:
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))

            cur_sim_data_09h[0] = (cur_sim_data_09h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_09h[1] = (cur_sim_data_09h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_09h[2] = (cur_sim_data_09h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_09h[3] = (cur_sim_data_09h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_09h[5] = (cur_sim_data_09h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]

            cur_sim_data_09h = np.moveaxis(cur_sim_data_09h, 0, -1)
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2): t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 2] = torch.FloatTensor(cur_sim_data_09h)

            cur_sim_f_15h_name = f"{self.sim_data_path}/{date_15h.year}/" + date_15h.strftime("%m%d") + f"15_{lead_from_15h:02d}.npy"
            if not os.path.exists(cur_sim_f_15h_name):
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_15h = np.load(cur_sim_f_15h_name)
            if len(cur_sim_data_15h.shape) != 3:
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            
            cur_sim_data_15h[0] = (cur_sim_data_15h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_15h[1] = (cur_sim_data_15h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_15h[2] = (cur_sim_data_15h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_15h[3] = (cur_sim_data_15h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_15h[5] = (cur_sim_data_15h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]

            cur_sim_data_15h = np.moveaxis(cur_sim_data_15h, 0, -1)
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 2: t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 3] = torch.FloatTensor(cur_sim_data_15h)

            cur_sim_f_21h_name = f"{self.sim_data_path}/{date_21h.year}/" + date_21h.strftime("%m%d") + f"21_{lead_from_21h:02d}.npy"
            if not os.path.exists(cur_sim_f_21h_name):
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_21h = np.load(cur_sim_f_21h_name)
            if len(cur_sim_data_21h.shape) != 3:
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            
            cur_sim_data_21h[0] = (cur_sim_data_21h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_21h[1] = (cur_sim_data_21h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_21h[2] = (cur_sim_data_21h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_21h[3] = (cur_sim_data_21h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_21h[5] = (cur_sim_data_21h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_sim_data_21h = np.moveaxis(cur_sim_data_21h, 0, -1)
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 3: t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4] = torch.FloatTensor(cur_sim_data_21h)
            

        range_4class = [(-1,15),(15,35),(35,75), (75,np.Inf)]
        class_four = [0,1,2,3]
        reanalysis_class = assign_class(reanalysis.numpy(), range_4class, class_four)

        return feats, masks, simulation, reanalysis, torch.IntTensor(reanalysis_class), torch.FloatTensor(raw_times), torch.FloatTensor(prev_pm25_vals)

    def collate_fn(self, samples):
        feats, masks, simulation, reanalysis, reanalysis_class, raw_times, prev_vals = zip(*samples)
        feats = torch.stack(feats, dim=0)
        masks = torch.stack(masks, dim=0)
        simulation = torch.stack(simulation, dim=0)
        reanalysis = torch.stack(reanalysis, dim=0)
        reanalysis_class = torch.stack(reanalysis_class, dim=0)
        raw_times = torch.stack(raw_times)
        prev_vals = torch.stack(prev_vals, dim=0)
        return feats, masks, simulation, reanalysis, reanalysis_class, raw_times, prev_vals

class Air_Simulation_Reanalysis_Dataset_v3(torch.utils.data.Dataset):
    """
    :param times: list of time in the dataset
    :param time_index: index of time in the dataset
    :param sat_outputs: satellite predictions
    :param sat_inputs: satellite observations
    :param feats,masks: features and corresponding masks
    :param input_dim: length of time series input
    :param output_dim: length of time series output
    :param prev_len: length of previous values for normalization
    :param korea_stn_num,china_stn_num: number of stations in Korea and China
    """
    def __init__(self, times, feats, masks, input_dim, output_dim, prev_len, korea_stn_num, china_stn_num, cmaq_size, sim_data_path, reanalysis_data_path, feat_infos):
        
        self.times = times
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prev_len = prev_len
        self.korea_stn_num = korea_stn_num
        self.china_stn_num = china_stn_num
        self.total_stn_num = korea_stn_num + china_stn_num
        self.cmaq_size = cmaq_size
        self.sim_data_path = sim_data_path
        self.reanalysis_data_path = reanalysis_data_path
        self.feat_infos = feat_infos
        
        self.feat_torch = torch.FloatTensor(feats)
        self.mask_torch = torch.from_numpy(masks)

    
    def load_feats(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        feats = self.feat_torch[mod_idx - self.input_dim + 1: mod_idx + 1]
        return feats
    
    def load_masks(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        masks = self.mask_torch[mod_idx - self.input_dim + 1: mod_idx + self.output_dim + 1].type(torch.BoolTensor)
        return masks
    
    def __len__(self) -> int:
        return (len(self.times) - (self.prev_len - 1) - self.output_dim)
    
    def __getitem__(self, idx):

        mod_idx = idx + (self.prev_len - 1)

        feats = self.load_feats(idx)
        masks = self.load_masks(idx)

        raw_times = []

        feat_dim = self.feat_torch.shape[-1]

        for t_idx in range(self.input_dim + self.output_dim):
            cur_idx = mod_idx - self.input_dim + 1 + t_idx
            raw_times.append([self.times[cur_idx].year, self.times[cur_idx].month, self.times[cur_idx].day, self.times[cur_idx].hour])

        simulation = torch.zeros(self.cmaq_size[0], self.cmaq_size[1], (self.input_dim + self.output_dim) * ((feat_dim // 2) * 4 + 4))
        reanalysis = torch.zeros(self.output_dim, self.cmaq_size[0], self.cmaq_size[1])
        curr_reanalysis = torch.zeros(self.cmaq_size[0], self.cmaq_size[1])

        cur_t_utc = self.times[mod_idx] - timedelta(hours=9)
        cur_f_name = f"{self.reanalysis_data_path}/{cur_t_utc.year}/ACONC.PM_RQ40i8a.KNU_09_01.{cur_t_utc.strftime('%Y%m%d')}.nc"
        ds = xr.open_dataset(cur_f_name)
        cur_pm25 = ds['PM2P5'].values
        cur_pm25 = cur_pm25[cur_t_utc.hour, 0]
        curr_reanalysis = torch.FloatTensor(cur_pm25)

        prev_pm25_vals = torch.zeros(self.prev_len, self.cmaq_size[0], self.cmaq_size[1])

        for t_idx in range(self.output_dim):
            cur_t_utc = self.times[mod_idx] + timedelta(hours=t_idx + 1) - timedelta(hours=9)
            cur_f_name = f"{self.reanalysis_data_path}/{cur_t_utc.year}/ACONC.PM_RQ40i8a.KNU_09_01.{cur_t_utc.strftime('%Y%m%d')}.nc"
            ds = xr.open_dataset(cur_f_name)
            cur_pm25 = ds['PM2P5'].values
            cur_pm25 = cur_pm25[cur_t_utc.hour, 0]
            reanalysis[t_idx] = torch.FloatTensor(cur_pm25)

        for t_idx in range(self.prev_len - self.input_dim):
            cur_idx = idx + t_idx
            cur_t_utc = self.times[cur_idx] - timedelta(hours=9)
            lead_from_03h = cur_t_utc.hour + 21
            if lead_from_03h >= 24:
                date_03h = cur_t_utc - timedelta(days=1)
            else:
                date_03h = cur_t_utc - timedelta(days=2)
                lead_from_03h += 24
            lead_from_09h = cur_t_utc.hour + 15
            if lead_from_09h >= 18:
                date_09h = cur_t_utc - timedelta(days=1)
            else:
                date_09h = cur_t_utc - timedelta(days=2)
                lead_from_09h += 24
            lead_from_15h = cur_t_utc.hour + 9
            if lead_from_15h >= 12:
                date_15h = cur_t_utc - timedelta(days=1)
            else:
                date_15h = cur_t_utc - timedelta(days=2)
                lead_from_15h += 24
            lead_from_21h = cur_t_utc.hour + 3
            if lead_from_21h >= 6:
                date_21h = cur_t_utc - timedelta(days=1)
            else:
                date_21h = cur_t_utc - timedelta(days=2)
                lead_from_21h += 24

            cur_sim_f_03h_name = f"{self.sim_data_path}/{date_03h.year}/" + date_03h.strftime("%m%d") + f"03_{lead_from_03h:02d}.npy"
            if not os.path.exists(cur_sim_f_03h_name):
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_03h = np.load(cur_sim_f_03h_name)
            if len(cur_sim_data_03h.shape) != 3:
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            cur_pm_25_03h = cur_sim_data_03h[4]

            cur_sim_f_09h_name = f"{self.sim_data_path}/{date_09h.year}/" + date_09h.strftime("%m%d") + f"09_{lead_from_09h:02d}.npy"
            if not os.path.exists(cur_sim_f_09h_name):
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_09h = np.load(cur_sim_f_09h_name)
            if len(cur_sim_data_09h.shape) != 3:
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            cur_pm_25_09h = cur_sim_data_09h[4]

            cur_sim_f_15h_name = f"{self.sim_data_path}/{date_15h.year}/" + date_15h.strftime("%m%d") + f"15_{lead_from_15h:02d}.npy"
            if not os.path.exists(cur_sim_f_15h_name):
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_15h = np.load(cur_sim_f_15h_name)
            if len(cur_sim_data_15h.shape) != 3:
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            cur_pm_25_15h = cur_sim_data_15h[4]
            
            cur_sim_f_21h_name = f"{self.sim_data_path}/{date_21h.year}/" + date_21h.strftime("%m%d") + f"21_{lead_from_21h:02d}.npy"
            if not os.path.exists(cur_sim_f_21h_name):
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_21h = np.load(cur_sim_f_21h_name)
            if len(cur_sim_data_21h.shape) != 3:
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            cur_pm_25_21h = cur_sim_data_21h[4]

            prev_pm25_vals[t_idx, :, :] = torch.FloatTensor(np.mean([cur_pm_25_03h, cur_pm_25_09h, cur_pm_25_15h, cur_pm_25_21h], axis=0))
        
        for t_idx in range(self.input_dim):
            cur_idx = mod_idx - self.input_dim + 1 + t_idx
            cur_t_utc = self.times[cur_idx] - timedelta(hours=9)
            lead_from_03h = cur_t_utc.hour + 21
            if lead_from_03h >= 24:
                date_03h = cur_t_utc - timedelta(days=1)
            else:
                date_03h = cur_t_utc - timedelta(days=2)
                lead_from_03h += 24
            lead_from_09h = cur_t_utc.hour + 15
            if lead_from_09h >= 18:
                date_09h = cur_t_utc - timedelta(days=1)
            else:
                date_09h = cur_t_utc - timedelta(days=2)
                lead_from_09h += 24
            lead_from_15h = cur_t_utc.hour + 9
            if lead_from_15h >= 12:
                date_15h = cur_t_utc - timedelta(days=1)
            else:
                date_15h = cur_t_utc - timedelta(days=2)
                lead_from_15h += 24
            lead_from_21h = cur_t_utc.hour + 3
            if lead_from_21h >= 6:
                date_21h = cur_t_utc - timedelta(days=1)
            else:
                date_21h = cur_t_utc - timedelta(days=2)
                lead_from_21h += 24
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4] = lead_from_03h
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 1] = lead_from_09h
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 2] = lead_from_15h
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 3] = lead_from_21h

            cur_sim_f_03h_name = f"{self.sim_data_path}/{date_03h.year}/" + date_03h.strftime("%m%d") + f"03_{lead_from_03h:02d}.npy"
            if not os.path.exists(cur_sim_f_03h_name):
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_03h = np.load(cur_sim_f_03h_name)
            if len(cur_sim_data_03h.shape) != 3:
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))

            cur_sim_data_03h[0] = (cur_sim_data_03h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_03h[1] = (cur_sim_data_03h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_03h[2] = (cur_sim_data_03h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_03h[3] = (cur_sim_data_03h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_03h[5] = (cur_sim_data_03h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_pm_25_03h = cur_sim_data_03h[4]

            cur_sim_data_03h = np.moveaxis(cur_sim_data_03h, 0, -1)
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4): t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2)] = torch.FloatTensor(cur_sim_data_03h)

            cur_sim_f_09h_name = f"{self.sim_data_path}/{date_09h.year}/" + date_09h.strftime("%m%d") + f"09_{lead_from_09h:02d}.npy"
            if not os.path.exists(cur_sim_f_09h_name):
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_09h = np.load(cur_sim_f_09h_name)
            if len(cur_sim_data_09h.shape) != 3:
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))

            cur_sim_data_09h[0] = (cur_sim_data_09h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_09h[1] = (cur_sim_data_09h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_09h[2] = (cur_sim_data_09h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_09h[3] = (cur_sim_data_09h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_09h[5] = (cur_sim_data_09h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_pm_25_09h = cur_sim_data_09h[4]

            cur_sim_data_09h = np.moveaxis(cur_sim_data_09h, 0, -1)
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2): t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 2] = torch.FloatTensor(cur_sim_data_09h)

            cur_sim_f_15h_name = f"{self.sim_data_path}/{date_15h.year}/" + date_15h.strftime("%m%d") + f"15_{lead_from_15h:02d}.npy"
            if not os.path.exists(cur_sim_f_15h_name):
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_15h = np.load(cur_sim_f_15h_name)
            if len(cur_sim_data_15h.shape) != 3:
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            
            cur_sim_data_15h[0] = (cur_sim_data_15h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_15h[1] = (cur_sim_data_15h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_15h[2] = (cur_sim_data_15h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_15h[3] = (cur_sim_data_15h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_15h[5] = (cur_sim_data_15h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_pm_25_15h = cur_sim_data_15h[4]

            cur_sim_data_15h = np.moveaxis(cur_sim_data_15h, 0, -1)
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 2: t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 3] = torch.FloatTensor(cur_sim_data_15h)

            cur_sim_f_21h_name = f"{self.sim_data_path}/{date_21h.year}/" + date_21h.strftime("%m%d") + f"21_{lead_from_21h:02d}.npy"
            if not os.path.exists(cur_sim_f_21h_name):
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_21h = np.load(cur_sim_f_21h_name)
            if len(cur_sim_data_21h.shape) != 3:
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            
            cur_sim_data_21h[0] = (cur_sim_data_21h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_21h[1] = (cur_sim_data_21h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_21h[2] = (cur_sim_data_21h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_21h[3] = (cur_sim_data_21h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_21h[5] = (cur_sim_data_21h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_pm_25_21h = cur_sim_data_21h[4]

            cur_sim_data_21h = np.moveaxis(cur_sim_data_21h, 0, -1)
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 3: t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4] = torch.FloatTensor(cur_sim_data_21h)

            prev_pm25_vals[t_idx + (self.prev_len - self.input_dim), :, :] = torch.FloatTensor(np.mean([cur_pm_25_03h, cur_pm_25_09h, cur_pm_25_15h, cur_pm_25_21h], axis=0))
        

        for t_idx in range(self.output_dim):

            cur_idx = mod_idx + t_idx + 1
            cur_t_utc = self.times[cur_idx] - timedelta(hours=9)
            lead_from_03h = cur_t_utc.hour + 21
            if lead_from_03h >= 24:
                date_03h = cur_t_utc - timedelta(days=1)
            else:
                date_03h = cur_t_utc - timedelta(days=2)
                lead_from_03h += 24
            lead_from_09h = cur_t_utc.hour + 15
            if lead_from_09h >= 18:
                date_09h = cur_t_utc - timedelta(days=1)
            else:
                date_09h = cur_t_utc - timedelta(days=2)
                lead_from_09h += 24
            lead_from_15h = cur_t_utc.hour + 9
            if lead_from_15h >= 12:
                date_15h = cur_t_utc - timedelta(days=1)
            else:
                date_15h = cur_t_utc - timedelta(days=2)
                lead_from_15h += 24
            lead_from_21h = cur_t_utc.hour + 3
            if lead_from_21h >= 6:
                date_21h = cur_t_utc - timedelta(days=1)
            else:
                date_21h = cur_t_utc - timedelta(days=2)
                lead_from_21h += 24
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4] = lead_from_03h
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 1] = lead_from_09h
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 2] = lead_from_15h
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 3] = lead_from_21h

            cur_sim_f_03h_name = f"{self.sim_data_path}/{date_03h.year}/" + date_03h.strftime("%m%d") + f"03_{lead_from_03h:02d}.npy"
            if not os.path.exists(cur_sim_f_03h_name):
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_03h = np.load(cur_sim_f_03h_name)
            if len(cur_sim_data_03h.shape) != 3:
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))

            cur_sim_data_03h[0] = (cur_sim_data_03h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_03h[1] = (cur_sim_data_03h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_03h[2] = (cur_sim_data_03h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_03h[3] = (cur_sim_data_03h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_03h[5] = (cur_sim_data_03h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]

            cur_sim_data_03h = np.moveaxis(cur_sim_data_03h, 0, -1)
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4): (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2)] = torch.FloatTensor(cur_sim_data_03h)

            cur_sim_f_09h_name = f"{self.sim_data_path}/{date_09h.year}/" + date_09h.strftime("%m%d") + f"09_{lead_from_09h:02d}.npy"
            if not os.path.exists(cur_sim_f_09h_name):
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_09h = np.load(cur_sim_f_09h_name)
            if len(cur_sim_data_09h.shape) != 3:
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))

            cur_sim_data_09h[0] = (cur_sim_data_09h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_09h[1] = (cur_sim_data_09h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_09h[2] = (cur_sim_data_09h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_09h[3] = (cur_sim_data_09h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_09h[5] = (cur_sim_data_09h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]

            cur_sim_data_09h = np.moveaxis(cur_sim_data_09h, 0, -1)
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2): (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 2] = torch.FloatTensor(cur_sim_data_09h)

            cur_sim_f_15h_name = f"{self.sim_data_path}/{date_15h.year}/" + date_15h.strftime("%m%d") + f"15_{lead_from_15h:02d}.npy"
            if not os.path.exists(cur_sim_f_15h_name):
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_15h = np.load(cur_sim_f_15h_name)
            if len(cur_sim_data_15h.shape) != 3:
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            
            cur_sim_data_15h[0] = (cur_sim_data_15h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_15h[1] = (cur_sim_data_15h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_15h[2] = (cur_sim_data_15h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_15h[3] = (cur_sim_data_15h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_15h[5] = (cur_sim_data_15h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]

            cur_sim_data_15h = np.moveaxis(cur_sim_data_15h, 0, -1)
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 2: (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 3] = torch.FloatTensor(cur_sim_data_15h)

            cur_sim_f_21h_name = f"{self.sim_data_path}/{date_21h.year}/" + date_21h.strftime("%m%d") + f"21_{lead_from_21h:02d}.npy"
            if not os.path.exists(cur_sim_f_21h_name):
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_21h = np.load(cur_sim_f_21h_name)
            if len(cur_sim_data_21h.shape) != 3:
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            
            cur_sim_data_21h[0] = (cur_sim_data_21h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_21h[1] = (cur_sim_data_21h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_21h[2] = (cur_sim_data_21h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_21h[3] = (cur_sim_data_21h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_21h[5] = (cur_sim_data_21h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_sim_data_21h = np.moveaxis(cur_sim_data_21h, 0, -1)
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 3: (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4] = torch.FloatTensor(cur_sim_data_21h)


        range_4class = [(-1,15),(15,35),(35,75), (75,np.Inf)]
        class_four = [0,1,2,3]
        reanalysis_class = assign_class(reanalysis.numpy(), range_4class, class_four)

        return feats, masks, simulation, curr_reanalysis, reanalysis, torch.IntTensor(reanalysis_class), torch.FloatTensor(raw_times), prev_pm25_vals

    def collate_fn(self, samples):
        feats, masks, simulation, curr_reanalysis, reanalysis, reanalysis_class, raw_times, prev_vals = zip(*samples)
        feats = torch.stack(feats, dim=0)
        masks = torch.stack(masks, dim=0)
        simulation = torch.stack(simulation, dim=0)
        curr_reanalysis = torch.stack(curr_reanalysis, dim=0)
        reanalysis = torch.stack(reanalysis, dim=0)
        reanalysis_class = torch.stack(reanalysis_class, dim=0)
        raw_times = torch.stack(raw_times)
        prev_vals = torch.stack(prev_vals, dim=0)
        return feats, masks, simulation, curr_reanalysis, reanalysis, reanalysis_class, raw_times, prev_vals












class Air_Simulation_Reanalysis_Dataset_only(torch.utils.data.Dataset):
    """
    :param times: list of time in the dataset
    :param time_index: index of time in the dataset
    :param sat_outputs: satellite predictions
    :param sat_inputs: satellite observations
    :param feats,masks: features and corresponding masks
    :param input_dim: length of time series input
    :param output_dim: length of time series output
    :param prev_len: length of previous values for normalization
    :param korea_stn_num,china_stn_num: number of stations in Korea and China
    """
    def __init__(self, times, feats, masks, input_dim, output_dim, prev_len, korea_stn_num, china_stn_num, cmaq_size, sim_data_path, reanalysis_data_path, feat_infos):
        
        self.times = times
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prev_len = prev_len
        self.korea_stn_num = korea_stn_num
        self.china_stn_num = china_stn_num
        self.total_stn_num = korea_stn_num + china_stn_num
        self.cmaq_size = cmaq_size
        self.sim_data_path = sim_data_path
        self.reanalysis_data_path = reanalysis_data_path
        # self.data_path = data_path
        self.feat_infos = feat_infos
        
        self.feat_torch = torch.FloatTensor(feats)
        self.mask_torch = torch.from_numpy(masks)

    
    def load_feats(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        feats = self.feat_torch[mod_idx - self.input_dim + 1: mod_idx + 1]
        return feats
    
    def load_masks(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        masks = self.mask_torch[mod_idx - self.input_dim + 1: mod_idx + self.output_dim + 1].type(torch.BoolTensor)
        return masks
    
    def __len__(self) -> int:
        return (len(self.times) - (self.prev_len - 1) - self.output_dim)
    
    def __getitem__(self, idx):

        mod_idx = idx + (self.prev_len - 1)

        feats = self.load_feats(idx)
        masks = self.load_masks(idx)

        raw_times = []

        feat_dim = self.feat_torch.shape[-1]

        for t_idx in range(self.input_dim + self.output_dim):
            cur_idx = mod_idx - self.input_dim + 1 + t_idx
            raw_times.append([self.times[cur_idx].year, self.times[cur_idx].month, self.times[cur_idx].day, self.times[cur_idx].hour])

        simulation = torch.zeros(self.cmaq_size[0], self.cmaq_size[1], (self.input_dim + self.output_dim) * ((feat_dim // 2) * 4 + 4))
        reanalysis = torch.zeros(self.output_dim, self.cmaq_size[0], self.cmaq_size[1])
        curr_reanalysis = torch.zeros(self.cmaq_size[0], self.cmaq_size[1])

        cur_t_utc = self.times[mod_idx] - timedelta(hours=9)
        cur_f_name = f"{self.reanalysis_data_path}/{cur_t_utc.year}/ACONC.PM_RQ40i8a.KNU_09_01.{cur_t_utc.strftime('%Y%m%d')}.nc"
        ds = xr.open_dataset(cur_f_name)
        cur_pm25 = ds['PM2P5'].values
        cur_pm25 = cur_pm25[cur_t_utc.hour, 0]
        curr_reanalysis = torch.FloatTensor(cur_pm25)

        prev_pm25_vals = torch.zeros(self.prev_len, self.cmaq_size[0], self.cmaq_size[1])

        for t_idx in range(self.output_dim):
            cur_t_utc = self.times[mod_idx] + timedelta(hours=t_idx + 1) - timedelta(hours=9)
            cur_f_name = f"{self.reanalysis_data_path}/{cur_t_utc.year}/ACONC.PM_RQ40i8a.KNU_09_01.{cur_t_utc.strftime('%Y%m%d')}.nc"
            ds = xr.open_dataset(cur_f_name)
            cur_pm25 = ds['PM2P5'].values
            cur_pm25 = cur_pm25[cur_t_utc.hour, 0]
            reanalysis[t_idx] = torch.FloatTensor(cur_pm25)

        for t_idx in range(self.prev_len - self.input_dim):
            cur_idx = idx + t_idx
            cur_t_utc = self.times[cur_idx] - timedelta(hours=9)
            lead_from_03h = cur_t_utc.hour + 21
            if lead_from_03h >= 24:
                date_03h = cur_t_utc - timedelta(days=1)
            else:
                date_03h = cur_t_utc - timedelta(days=2)
                lead_from_03h += 24
            lead_from_09h = cur_t_utc.hour + 15
            if lead_from_09h >= 18:
                date_09h = cur_t_utc - timedelta(days=1)
            else:
                date_09h = cur_t_utc - timedelta(days=2)
                lead_from_09h += 24
            lead_from_15h = cur_t_utc.hour + 9
            if lead_from_15h >= 12:
                date_15h = cur_t_utc - timedelta(days=1)
            else:
                date_15h = cur_t_utc - timedelta(days=2)
                lead_from_15h += 24
            lead_from_21h = cur_t_utc.hour + 3
            if lead_from_21h >= 6:
                date_21h = cur_t_utc - timedelta(days=1)
            else:
                date_21h = cur_t_utc - timedelta(days=2)
                lead_from_21h += 24

            cur_sim_f_03h_name = f"{self.sim_data_path}/{date_03h.year}/" + date_03h.strftime("%m%d") + f"03_{lead_from_03h:02d}.npy"
            if not os.path.exists(cur_sim_f_03h_name):
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_03h = np.load(cur_sim_f_03h_name)
            if len(cur_sim_data_03h.shape) != 3:
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            cur_pm_25_03h = cur_sim_data_03h[4]

            cur_sim_f_09h_name = f"{self.sim_data_path}/{date_09h.year}/" + date_09h.strftime("%m%d") + f"09_{lead_from_09h:02d}.npy"
            if not os.path.exists(cur_sim_f_09h_name):
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_09h = np.load(cur_sim_f_09h_name)
            if len(cur_sim_data_09h.shape) != 3:
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            cur_pm_25_09h = cur_sim_data_09h[4]

            cur_sim_f_15h_name = f"{self.sim_data_path}/{date_15h.year}/" + date_15h.strftime("%m%d") + f"15_{lead_from_15h:02d}.npy"
            if not os.path.exists(cur_sim_f_15h_name):
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_15h = np.load(cur_sim_f_15h_name)
            if len(cur_sim_data_15h.shape) != 3:
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            cur_pm_25_15h = cur_sim_data_15h[4]
            
            cur_sim_f_21h_name = f"{self.sim_data_path}/{date_21h.year}/" + date_21h.strftime("%m%d") + f"21_{lead_from_21h:02d}.npy"
            if not os.path.exists(cur_sim_f_21h_name):
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_21h = np.load(cur_sim_f_21h_name)
            if len(cur_sim_data_21h.shape) != 3:
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            cur_pm_25_21h = cur_sim_data_21h[4]

            prev_pm25_vals[t_idx, :, :] = torch.FloatTensor(np.mean([cur_pm_25_03h, cur_pm_25_09h, cur_pm_25_15h, cur_pm_25_21h], axis=0))
        
        for t_idx in range(self.input_dim):
            cur_idx = mod_idx - self.input_dim + 1 + t_idx
            cur_t_utc = self.times[cur_idx] - timedelta(hours=9)
            lead_from_03h = cur_t_utc.hour + 21
            if lead_from_03h >= 24:
                date_03h = cur_t_utc - timedelta(days=1)
            else:
                date_03h = cur_t_utc - timedelta(days=2)
                lead_from_03h += 24
            lead_from_09h = cur_t_utc.hour + 15
            if lead_from_09h >= 18:
                date_09h = cur_t_utc - timedelta(days=1)
            else:
                date_09h = cur_t_utc - timedelta(days=2)
                lead_from_09h += 24
            lead_from_15h = cur_t_utc.hour + 9
            if lead_from_15h >= 12:
                date_15h = cur_t_utc - timedelta(days=1)
            else:
                date_15h = cur_t_utc - timedelta(days=2)
                lead_from_15h += 24
            lead_from_21h = cur_t_utc.hour + 3
            if lead_from_21h >= 6:
                date_21h = cur_t_utc - timedelta(days=1)
            else:
                date_21h = cur_t_utc - timedelta(days=2)
                lead_from_21h += 24
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4] = lead_from_03h
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 1] = lead_from_09h
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 2] = lead_from_15h
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 3] = lead_from_21h

            cur_sim_f_03h_name = f"{self.sim_data_path}/{date_03h.year}/" + date_03h.strftime("%m%d") + f"03_{lead_from_03h:02d}.npy"
            if not os.path.exists(cur_sim_f_03h_name):
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_03h = np.load(cur_sim_f_03h_name)
            if len(cur_sim_data_03h.shape) != 3:
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))

            cur_sim_data_03h[0] = (cur_sim_data_03h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_03h[1] = (cur_sim_data_03h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_03h[2] = (cur_sim_data_03h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_03h[3] = (cur_sim_data_03h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_03h[5] = (cur_sim_data_03h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_pm_25_03h = cur_sim_data_03h[4]

            cur_sim_data_03h = np.moveaxis(cur_sim_data_03h, 0, -1)
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4): t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2)] = torch.FloatTensor(cur_sim_data_03h)

            cur_sim_f_09h_name = f"{self.sim_data_path}/{date_09h.year}/" + date_09h.strftime("%m%d") + f"09_{lead_from_09h:02d}.npy"
            if not os.path.exists(cur_sim_f_09h_name):
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_09h = np.load(cur_sim_f_09h_name)
            if len(cur_sim_data_09h.shape) != 3:
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))

            cur_sim_data_09h[0] = (cur_sim_data_09h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_09h[1] = (cur_sim_data_09h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_09h[2] = (cur_sim_data_09h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_09h[3] = (cur_sim_data_09h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_09h[5] = (cur_sim_data_09h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_pm_25_09h = cur_sim_data_09h[4]

            cur_sim_data_09h = np.moveaxis(cur_sim_data_09h, 0, -1)
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2): t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 2] = torch.FloatTensor(cur_sim_data_09h)

            cur_sim_f_15h_name = f"{self.sim_data_path}/{date_15h.year}/" + date_15h.strftime("%m%d") + f"15_{lead_from_15h:02d}.npy"
            if not os.path.exists(cur_sim_f_15h_name):
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_15h = np.load(cur_sim_f_15h_name)
            if len(cur_sim_data_15h.shape) != 3:
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            
            cur_sim_data_15h[0] = (cur_sim_data_15h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_15h[1] = (cur_sim_data_15h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_15h[2] = (cur_sim_data_15h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_15h[3] = (cur_sim_data_15h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_15h[5] = (cur_sim_data_15h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_pm_25_15h = cur_sim_data_15h[4]

            cur_sim_data_15h = np.moveaxis(cur_sim_data_15h, 0, -1)
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 2: t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 3] = torch.FloatTensor(cur_sim_data_15h)

            cur_sim_f_21h_name = f"{self.sim_data_path}/{date_21h.year}/" + date_21h.strftime("%m%d") + f"21_{lead_from_21h:02d}.npy"
            if not os.path.exists(cur_sim_f_21h_name):
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_21h = np.load(cur_sim_f_21h_name)
            if len(cur_sim_data_21h.shape) != 3:
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            
            cur_sim_data_21h[0] = (cur_sim_data_21h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_21h[1] = (cur_sim_data_21h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_21h[2] = (cur_sim_data_21h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_21h[3] = (cur_sim_data_21h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_21h[5] = (cur_sim_data_21h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_pm_25_21h = cur_sim_data_21h[4]

            cur_sim_data_21h = np.moveaxis(cur_sim_data_21h, 0, -1)
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 3: t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4] = torch.FloatTensor(cur_sim_data_21h)

            prev_pm25_vals[t_idx + (self.prev_len - self.input_dim), :, :] = torch.FloatTensor(np.mean([cur_pm_25_03h, cur_pm_25_09h, cur_pm_25_15h, cur_pm_25_21h], axis=0))
        

        for t_idx in range(self.output_dim):

            cur_idx = mod_idx + t_idx + 1
            cur_t_utc = self.times[cur_idx] - timedelta(hours=9)
            lead_from_03h = cur_t_utc.hour + 21
            if lead_from_03h >= 24:
                date_03h = cur_t_utc - timedelta(days=1)
            else:
                date_03h = cur_t_utc - timedelta(days=2)
                lead_from_03h += 24
            lead_from_09h = cur_t_utc.hour + 15
            if lead_from_09h >= 18:
                date_09h = cur_t_utc - timedelta(days=1)
            else:
                date_09h = cur_t_utc - timedelta(days=2)
                lead_from_09h += 24
            lead_from_15h = cur_t_utc.hour + 9
            if lead_from_15h >= 12:
                date_15h = cur_t_utc - timedelta(days=1)
            else:
                date_15h = cur_t_utc - timedelta(days=2)
                lead_from_15h += 24
            lead_from_21h = cur_t_utc.hour + 3
            if lead_from_21h >= 6:
                date_21h = cur_t_utc - timedelta(days=1)
            else:
                date_21h = cur_t_utc - timedelta(days=2)
                lead_from_21h += 24
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4] = lead_from_03h
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 1] = lead_from_09h
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 2] = lead_from_15h
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 3] = lead_from_21h

            cur_sim_f_03h_name = f"{self.sim_data_path}/{date_03h.year}/" + date_03h.strftime("%m%d") + f"03_{lead_from_03h:02d}.npy"
            if not os.path.exists(cur_sim_f_03h_name):
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_03h = np.load(cur_sim_f_03h_name)
            if len(cur_sim_data_03h.shape) != 3:
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))

            cur_sim_data_03h[0] = (cur_sim_data_03h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_03h[1] = (cur_sim_data_03h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_03h[2] = (cur_sim_data_03h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_03h[3] = (cur_sim_data_03h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_03h[5] = (cur_sim_data_03h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]

            cur_sim_data_03h = np.moveaxis(cur_sim_data_03h, 0, -1)
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4): (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2)] = torch.FloatTensor(cur_sim_data_03h)

            cur_sim_f_09h_name = f"{self.sim_data_path}/{date_09h.year}/" + date_09h.strftime("%m%d") + f"09_{lead_from_09h:02d}.npy"
            if not os.path.exists(cur_sim_f_09h_name):
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_09h = np.load(cur_sim_f_09h_name)
            if len(cur_sim_data_09h.shape) != 3:
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))

            cur_sim_data_09h[0] = (cur_sim_data_09h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_09h[1] = (cur_sim_data_09h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_09h[2] = (cur_sim_data_09h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_09h[3] = (cur_sim_data_09h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_09h[5] = (cur_sim_data_09h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]

            cur_sim_data_09h = np.moveaxis(cur_sim_data_09h, 0, -1)
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2): (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 2] = torch.FloatTensor(cur_sim_data_09h)

            cur_sim_f_15h_name = f"{self.sim_data_path}/{date_15h.year}/" + date_15h.strftime("%m%d") + f"15_{lead_from_15h:02d}.npy"
            if not os.path.exists(cur_sim_f_15h_name):
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_15h = np.load(cur_sim_f_15h_name)
            if len(cur_sim_data_15h.shape) != 3:
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            
            cur_sim_data_15h[0] = (cur_sim_data_15h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_15h[1] = (cur_sim_data_15h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_15h[2] = (cur_sim_data_15h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_15h[3] = (cur_sim_data_15h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_15h[5] = (cur_sim_data_15h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]

            cur_sim_data_15h = np.moveaxis(cur_sim_data_15h, 0, -1)
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 2: (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 3] = torch.FloatTensor(cur_sim_data_15h)

            cur_sim_f_21h_name = f"{self.sim_data_path}/{date_21h.year}/" + date_21h.strftime("%m%d") + f"21_{lead_from_21h:02d}.npy"
            if not os.path.exists(cur_sim_f_21h_name):
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_21h = np.load(cur_sim_f_21h_name)
            if len(cur_sim_data_21h.shape) != 3:
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            
            cur_sim_data_21h[0] = (cur_sim_data_21h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_21h[1] = (cur_sim_data_21h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_21h[2] = (cur_sim_data_21h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_21h[3] = (cur_sim_data_21h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_21h[5] = (cur_sim_data_21h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_sim_data_21h = np.moveaxis(cur_sim_data_21h, 0, -1)
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 3: (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4] = torch.FloatTensor(cur_sim_data_21h)


        range_4class = [(-1,15),(15,35),(35,75), (75,np.Inf)]
        class_four = [0,1,2,3]
        reanalysis_class = assign_class(reanalysis.numpy(), range_4class, class_four)

        return simulation, curr_reanalysis, reanalysis, torch.IntTensor(reanalysis_class), torch.FloatTensor(raw_times), prev_pm25_vals

    def collate_fn(self, samples):
        simulation, curr_reanalysis, reanalysis, reanalysis_class, raw_times, prev_vals = zip(*samples)
        # feats = torch.stack(feats, dim=0)
        # masks = torch.stack(masks, dim=0)
        simulation = torch.stack(simulation, dim=0)
        curr_reanalysis = torch.stack(curr_reanalysis, dim=0)
        reanalysis = torch.stack(reanalysis, dim=0)
        reanalysis_class = torch.stack(reanalysis_class, dim=0)
        raw_times = torch.stack(raw_times)
        prev_vals = torch.stack(prev_vals, dim=0)
        return simulation, curr_reanalysis, reanalysis, reanalysis_class, raw_times, prev_vals











class Air_Simulation_Reanalysis_Dataset_with_station_imgs(torch.utils.data.Dataset):
    """
    :param times: list of time in the dataset
    :param time_index: index of time in the dataset
    :param sat_outputs: satellite predictions
    :param sat_inputs: satellite observations
    :param feats,masks: features and corresponding masks
    :param input_dim: length of time series input
    :param output_dim: length of time series output
    :param prev_len: length of previous values for normalization
    :param korea_stn_num,china_stn_num: number of stations in Korea and China
    """
    def __init__(self, times, feats, masks, input_dim, output_dim, prev_len, korea_stn_num, china_stn_num, cmaq_size, sim_data_path, reanalysis_data_path, data_path, feat_infos):
        
        self.times = times
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prev_len = prev_len
        self.korea_stn_num = korea_stn_num
        self.china_stn_num = china_stn_num
        self.total_stn_num = korea_stn_num + china_stn_num
        self.cmaq_size = cmaq_size
        self.sim_data_path = sim_data_path
        self.reanalysis_data_path = reanalysis_data_path
        self.data_path = data_path
        self.feat_infos = feat_infos
        
        self.feat_torch = torch.FloatTensor(feats)
        self.mask_torch = torch.from_numpy(masks)

    
    def load_feats(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        feats = self.feat_torch[mod_idx - self.input_dim + 1: mod_idx + 1]
        return feats
    
    def load_masks(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        masks = self.mask_torch[mod_idx - self.input_dim + 1: mod_idx + self.output_dim + 1].type(torch.BoolTensor)
        return masks
    
    def __len__(self) -> int:
        return (len(self.times) - (self.prev_len - 1) - self.output_dim)
    
    def __getitem__(self, idx):

        mod_idx = idx + (self.prev_len - 1)

        feats = self.load_feats(idx)
        masks = self.load_masks(idx)

        raw_times = []

        feat_dim = self.feat_torch.shape[-1]

        for t_idx in range(self.input_dim + self.output_dim):
            cur_idx = mod_idx - self.input_dim + 1 + t_idx
            raw_times.append([self.times[cur_idx].year, self.times[cur_idx].month, self.times[cur_idx].day, self.times[cur_idx].hour])

        simulation = torch.zeros(self.cmaq_size[0], self.cmaq_size[1], (self.input_dim + self.output_dim) * ((feat_dim // 2) * 4 + 4))
        reanalysis = torch.zeros(self.output_dim, self.cmaq_size[0], self.cmaq_size[1])
        curr_reanalysis = torch.zeros(self.cmaq_size[0], self.cmaq_size[1])
        station_based_inputs = torch.zeros(self.input_dim, 2, self.cmaq_size[0], self.cmaq_size[1])
        station_based_multiair_outputs = torch.zeros(self.output_dim, 2, self.cmaq_size[0], self.cmaq_size[1])

        cur_t_utc = self.times[mod_idx] - timedelta(hours=9)
        cur_f_name = f"{self.reanalysis_data_path}/{cur_t_utc.year}/ACONC.PM_RQ40i8a.KNU_09_01.{cur_t_utc.strftime('%Y%m%d')}.nc"
        ds = xr.open_dataset(cur_f_name)
        cur_pm25 = ds['PM2P5'].values
        cur_pm25 = cur_pm25[cur_t_utc.hour, 0]
        curr_reanalysis = torch.FloatTensor(cur_pm25)

        prev_pm25_vals = torch.zeros(self.prev_len, self.cmaq_size[0], self.cmaq_size[1])

        for t_idx in range(self.output_dim):
            cur_t_utc = self.times[mod_idx] + timedelta(hours=t_idx + 1) - timedelta(hours=9)
            cur_f_name = f"{self.reanalysis_data_path}/{cur_t_utc.year}/ACONC.PM_RQ40i8a.KNU_09_01.{cur_t_utc.strftime('%Y%m%d')}.nc"
            ds = xr.open_dataset(cur_f_name)
            cur_pm25 = ds['PM2P5'].values
            cur_pm25 = cur_pm25[cur_t_utc.hour, 0]
            reanalysis[t_idx] = torch.FloatTensor(cur_pm25)

        for t_idx in range(self.prev_len - self.input_dim):
            cur_idx = idx + t_idx
            cur_t_utc = self.times[cur_idx] - timedelta(hours=9)
            lead_from_03h = cur_t_utc.hour + 21
            if lead_from_03h >= 24:
                date_03h = cur_t_utc - timedelta(days=1)
            else:
                date_03h = cur_t_utc - timedelta(days=2)
                lead_from_03h += 24
            lead_from_09h = cur_t_utc.hour + 15
            if lead_from_09h >= 18:
                date_09h = cur_t_utc - timedelta(days=1)
            else:
                date_09h = cur_t_utc - timedelta(days=2)
                lead_from_09h += 24
            lead_from_15h = cur_t_utc.hour + 9
            if lead_from_15h >= 12:
                date_15h = cur_t_utc - timedelta(days=1)
            else:
                date_15h = cur_t_utc - timedelta(days=2)
                lead_from_15h += 24
            lead_from_21h = cur_t_utc.hour + 3
            if lead_from_21h >= 6:
                date_21h = cur_t_utc - timedelta(days=1)
            else:
                date_21h = cur_t_utc - timedelta(days=2)
                lead_from_21h += 24

            cur_sim_f_03h_name = f"{self.sim_data_path}/{date_03h.year}/" + date_03h.strftime("%m%d") + f"03_{lead_from_03h:02d}.npy"
            if not os.path.exists(cur_sim_f_03h_name):
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_03h = np.load(cur_sim_f_03h_name)
            if len(cur_sim_data_03h.shape) != 3:
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            cur_pm_25_03h = cur_sim_data_03h[4]

            cur_sim_f_09h_name = f"{self.sim_data_path}/{date_09h.year}/" + date_09h.strftime("%m%d") + f"09_{lead_from_09h:02d}.npy"
            if not os.path.exists(cur_sim_f_09h_name):
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_09h = np.load(cur_sim_f_09h_name)
            if len(cur_sim_data_09h.shape) != 3:
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            cur_pm_25_09h = cur_sim_data_09h[4]

            cur_sim_f_15h_name = f"{self.sim_data_path}/{date_15h.year}/" + date_15h.strftime("%m%d") + f"15_{lead_from_15h:02d}.npy"
            if not os.path.exists(cur_sim_f_15h_name):
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_15h = np.load(cur_sim_f_15h_name)
            if len(cur_sim_data_15h.shape) != 3:
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            cur_pm_25_15h = cur_sim_data_15h[4]
            
            cur_sim_f_21h_name = f"{self.sim_data_path}/{date_21h.year}/" + date_21h.strftime("%m%d") + f"21_{lead_from_21h:02d}.npy"
            if not os.path.exists(cur_sim_f_21h_name):
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_21h = np.load(cur_sim_f_21h_name)
            if len(cur_sim_data_21h.shape) != 3:
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            cur_pm_25_21h = cur_sim_data_21h[4]

            prev_pm25_vals[t_idx, :, :] = torch.FloatTensor(np.mean([cur_pm_25_03h, cur_pm_25_09h, cur_pm_25_15h, cur_pm_25_21h], axis=0))
        
        for t_idx in range(self.input_dim):
            cur_idx = mod_idx - self.input_dim + 1 + t_idx
            curr_time = self.times[cur_idx]
            curr_station_img = np.load(f'{self.data_path}/ground_obs_imgs/{curr_time.strftime("%Y")}/{int(curr_time.strftime("%m"))}/{curr_time.strftime("%d%H")}_img.npy')
            curr_station_krig_img = np.load(f'{self.data_path}/ground_obs_krig_imgs/{curr_time.strftime("%Y")}/{int(curr_time.strftime("%m"))}/{curr_time.strftime("%d%H")}_krige_img.npy')
            # print(curr_station_img.shape, curr_station_krig_img.shape)
            # curr_station_input = np.concatenate([np.expand_dims(curr_station_img, axis=0), curr_station_krig_img], axis=0)
            station_based_inputs[t_idx] = torch.FloatTensor(curr_station_krig_img)

            cur_t_utc = self.times[cur_idx] - timedelta(hours=9)
            lead_from_03h = cur_t_utc.hour + 21
            if lead_from_03h >= 24:
                date_03h = cur_t_utc - timedelta(days=1)
            else:
                date_03h = cur_t_utc - timedelta(days=2)
                lead_from_03h += 24
            lead_from_09h = cur_t_utc.hour + 15
            if lead_from_09h >= 18:
                date_09h = cur_t_utc - timedelta(days=1)
            else:
                date_09h = cur_t_utc - timedelta(days=2)
                lead_from_09h += 24
            lead_from_15h = cur_t_utc.hour + 9
            if lead_from_15h >= 12:
                date_15h = cur_t_utc - timedelta(days=1)
            else:
                date_15h = cur_t_utc - timedelta(days=2)
                lead_from_15h += 24
            lead_from_21h = cur_t_utc.hour + 3
            if lead_from_21h >= 6:
                date_21h = cur_t_utc - timedelta(days=1)
            else:
                date_21h = cur_t_utc - timedelta(days=2)
                lead_from_21h += 24
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4] = lead_from_03h
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 1] = lead_from_09h
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 2] = lead_from_15h
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 3] = lead_from_21h

            cur_sim_f_03h_name = f"{self.sim_data_path}/{date_03h.year}/" + date_03h.strftime("%m%d") + f"03_{lead_from_03h:02d}.npy"
            if not os.path.exists(cur_sim_f_03h_name):
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_03h = np.load(cur_sim_f_03h_name)
            if len(cur_sim_data_03h.shape) != 3:
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))

            cur_sim_data_03h[0] = (cur_sim_data_03h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_03h[1] = (cur_sim_data_03h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_03h[2] = (cur_sim_data_03h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_03h[3] = (cur_sim_data_03h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_03h[5] = (cur_sim_data_03h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_pm_25_03h = cur_sim_data_03h[4]

            cur_sim_data_03h = np.moveaxis(cur_sim_data_03h, 0, -1)
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4): t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2)] = torch.FloatTensor(cur_sim_data_03h)

            cur_sim_f_09h_name = f"{self.sim_data_path}/{date_09h.year}/" + date_09h.strftime("%m%d") + f"09_{lead_from_09h:02d}.npy"
            if not os.path.exists(cur_sim_f_09h_name):
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_09h = np.load(cur_sim_f_09h_name)
            if len(cur_sim_data_09h.shape) != 3:
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))

            cur_sim_data_09h[0] = (cur_sim_data_09h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_09h[1] = (cur_sim_data_09h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_09h[2] = (cur_sim_data_09h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_09h[3] = (cur_sim_data_09h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_09h[5] = (cur_sim_data_09h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_pm_25_09h = cur_sim_data_09h[4]

            cur_sim_data_09h = np.moveaxis(cur_sim_data_09h, 0, -1)
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2): t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 2] = torch.FloatTensor(cur_sim_data_09h)

            cur_sim_f_15h_name = f"{self.sim_data_path}/{date_15h.year}/" + date_15h.strftime("%m%d") + f"15_{lead_from_15h:02d}.npy"
            if not os.path.exists(cur_sim_f_15h_name):
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_15h = np.load(cur_sim_f_15h_name)
            if len(cur_sim_data_15h.shape) != 3:
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            
            cur_sim_data_15h[0] = (cur_sim_data_15h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_15h[1] = (cur_sim_data_15h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_15h[2] = (cur_sim_data_15h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_15h[3] = (cur_sim_data_15h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_15h[5] = (cur_sim_data_15h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_pm_25_15h = cur_sim_data_15h[4]

            cur_sim_data_15h = np.moveaxis(cur_sim_data_15h, 0, -1)
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 2: t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 3] = torch.FloatTensor(cur_sim_data_15h)

            cur_sim_f_21h_name = f"{self.sim_data_path}/{date_21h.year}/" + date_21h.strftime("%m%d") + f"21_{lead_from_21h:02d}.npy"
            if not os.path.exists(cur_sim_f_21h_name):
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_21h = np.load(cur_sim_f_21h_name)
            if len(cur_sim_data_21h.shape) != 3:
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            
            cur_sim_data_21h[0] = (cur_sim_data_21h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_21h[1] = (cur_sim_data_21h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_21h[2] = (cur_sim_data_21h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_21h[3] = (cur_sim_data_21h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_21h[5] = (cur_sim_data_21h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_pm_25_21h = cur_sim_data_21h[4]

            cur_sim_data_21h = np.moveaxis(cur_sim_data_21h, 0, -1)
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 3: t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4] = torch.FloatTensor(cur_sim_data_21h)

            prev_pm25_vals[t_idx + (self.prev_len - self.input_dim), :, :] = torch.FloatTensor(np.mean([cur_pm_25_03h, cur_pm_25_09h, cur_pm_25_15h, cur_pm_25_21h], axis=0))
        
        curr_time_for_outputs = self.times[mod_idx]
        multiair_pred_img = np.load(f'{self.data_path}/multiair_img/{curr_time_for_outputs.strftime("%Y")}/{int(curr_time_for_outputs.strftime("%m"))}/{curr_time_for_outputs.strftime("%d%H")}_multiair_img.npy')
        multiair_pred_krig_img = np.load(f'{self.data_path}/multiair_krig_img/{curr_time_for_outputs.strftime("%Y")}/{int(curr_time_for_outputs.strftime("%m"))}/{curr_time_for_outputs.strftime("%d%H")}_multiair_krige_img.npy')
        # print(np.expand_dims(multiair_pred_img, axis=1).shape, multiair_pred_krig_img.shape)
        # curr_station_input = np.concatenate([np.expand_dims(multiair_pred_img, axis=1), multiair_pred_krig_img], axis=1)
        station_based_multiair_outputs = torch.FloatTensor(multiair_pred_krig_img)[:self.output_dim,:,:,:]
        for t_idx in range(self.output_dim):

            cur_idx = mod_idx + t_idx + 1
            cur_t_utc = self.times[cur_idx] - timedelta(hours=9)
            lead_from_03h = cur_t_utc.hour + 21
            if lead_from_03h >= 24:
                date_03h = cur_t_utc - timedelta(days=1)
            else:
                date_03h = cur_t_utc - timedelta(days=2)
                lead_from_03h += 24
            lead_from_09h = cur_t_utc.hour + 15
            if lead_from_09h >= 18:
                date_09h = cur_t_utc - timedelta(days=1)
            else:
                date_09h = cur_t_utc - timedelta(days=2)
                lead_from_09h += 24
            lead_from_15h = cur_t_utc.hour + 9
            if lead_from_15h >= 12:
                date_15h = cur_t_utc - timedelta(days=1)
            else:
                date_15h = cur_t_utc - timedelta(days=2)
                lead_from_15h += 24
            lead_from_21h = cur_t_utc.hour + 3
            if lead_from_21h >= 6:
                date_21h = cur_t_utc - timedelta(days=1)
            else:
                date_21h = cur_t_utc - timedelta(days=2)
                lead_from_21h += 24
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4] = lead_from_03h
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 1] = lead_from_09h
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 2] = lead_from_15h
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 3] = lead_from_21h

            cur_sim_f_03h_name = f"{self.sim_data_path}/{date_03h.year}/" + date_03h.strftime("%m%d") + f"03_{lead_from_03h:02d}.npy"
            if not os.path.exists(cur_sim_f_03h_name):
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_03h = np.load(cur_sim_f_03h_name)
            if len(cur_sim_data_03h.shape) != 3:
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))

            cur_sim_data_03h[0] = (cur_sim_data_03h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_03h[1] = (cur_sim_data_03h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_03h[2] = (cur_sim_data_03h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_03h[3] = (cur_sim_data_03h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_03h[5] = (cur_sim_data_03h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]

            cur_sim_data_03h = np.moveaxis(cur_sim_data_03h, 0, -1)
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4): (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2)] = torch.FloatTensor(cur_sim_data_03h)

            cur_sim_f_09h_name = f"{self.sim_data_path}/{date_09h.year}/" + date_09h.strftime("%m%d") + f"09_{lead_from_09h:02d}.npy"
            if not os.path.exists(cur_sim_f_09h_name):
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_09h = np.load(cur_sim_f_09h_name)
            if len(cur_sim_data_09h.shape) != 3:
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))

            cur_sim_data_09h[0] = (cur_sim_data_09h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_09h[1] = (cur_sim_data_09h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_09h[2] = (cur_sim_data_09h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_09h[3] = (cur_sim_data_09h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_09h[5] = (cur_sim_data_09h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]

            cur_sim_data_09h = np.moveaxis(cur_sim_data_09h, 0, -1)
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2): (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 2] = torch.FloatTensor(cur_sim_data_09h)

            cur_sim_f_15h_name = f"{self.sim_data_path}/{date_15h.year}/" + date_15h.strftime("%m%d") + f"15_{lead_from_15h:02d}.npy"
            if not os.path.exists(cur_sim_f_15h_name):
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_15h = np.load(cur_sim_f_15h_name)
            if len(cur_sim_data_15h.shape) != 3:
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            
            cur_sim_data_15h[0] = (cur_sim_data_15h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_15h[1] = (cur_sim_data_15h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_15h[2] = (cur_sim_data_15h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_15h[3] = (cur_sim_data_15h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_15h[5] = (cur_sim_data_15h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]

            cur_sim_data_15h = np.moveaxis(cur_sim_data_15h, 0, -1)
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 2: (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 3] = torch.FloatTensor(cur_sim_data_15h)

            cur_sim_f_21h_name = f"{self.sim_data_path}/{date_21h.year}/" + date_21h.strftime("%m%d") + f"21_{lead_from_21h:02d}.npy"
            if not os.path.exists(cur_sim_f_21h_name):
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_21h = np.load(cur_sim_f_21h_name)
            if len(cur_sim_data_21h.shape) != 3:
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            
            cur_sim_data_21h[0] = (cur_sim_data_21h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_21h[1] = (cur_sim_data_21h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_21h[2] = (cur_sim_data_21h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_21h[3] = (cur_sim_data_21h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_21h[5] = (cur_sim_data_21h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_sim_data_21h = np.moveaxis(cur_sim_data_21h, 0, -1)
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 3: (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4] = torch.FloatTensor(cur_sim_data_21h)


        range_4class = [(-1,15),(15,35),(35,75), (75,np.Inf)]
        class_four = [0,1,2,3]
        reanalysis_class = assign_class(reanalysis.numpy(), range_4class, class_four)

        return simulation, curr_reanalysis, reanalysis, torch.IntTensor(reanalysis_class), torch.FloatTensor(raw_times), prev_pm25_vals, station_based_inputs, station_based_multiair_outputs

    def collate_fn(self, samples):
        simulation, curr_reanalysis, reanalysis, reanalysis_class, raw_times, prev_vals, station_based_inputs, station_based_multiair_outputs = zip(*samples)
        # feats = torch.stack(feats, dim=0)
        # masks = torch.stack(masks, dim=0)
        simulation = torch.stack(simulation, dim=0)
        curr_reanalysis = torch.stack(curr_reanalysis, dim=0)
        reanalysis = torch.stack(reanalysis, dim=0)
        reanalysis_class = torch.stack(reanalysis_class, dim=0)
        raw_times = torch.stack(raw_times)
        prev_vals = torch.stack(prev_vals, dim=0)
        station_based_inputs = torch.stack(station_based_inputs, dim=0)
        station_based_multiair_outputs = torch.stack(station_based_multiair_outputs, dim=0)
        return simulation, curr_reanalysis, reanalysis, reanalysis_class, raw_times, prev_vals, station_based_inputs, station_based_multiair_outputs






class Air_Simulation_Reanalysis_Dataset_by_stn(torch.utils.data.Dataset):
    """
    :param times: list of time in the dataset
    :param time_index: index of time in the dataset
    :param sat_outputs: satellite predictions
    :param sat_inputs: satellite observations
    :param feats,masks: features and corresponding masks
    :param input_dim: length of time series input
    :param output_dim: length of time series output
    :param prev_len: length of previous values for normalization
    :param korea_stn_num,china_stn_num: number of stations in Korea and China
    """
    def __init__(self, times, feats, masks, input_dim, output_dim, prev_len, korea_stn_num, china_stn_num, cmaq_size, sim_data_path, reanalysis_data_path, feat_infos):
        
        self.times = times
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prev_len = prev_len
        self.korea_stn_num = korea_stn_num
        self.china_stn_num = china_stn_num
        self.total_stn_num = korea_stn_num + china_stn_num
        self.cmaq_size = cmaq_size
        self.sim_data_path = sim_data_path
        self.reanalysis_data_path = reanalysis_data_path
        self.feat_infos = feat_infos
        
        self.feat_torch = torch.FloatTensor(feats)
        self.mask_torch = torch.from_numpy(masks)

    
    def load_feats(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        feats = self.feat_torch[mod_idx - self.input_dim + 1: mod_idx + 1]
        return feats
    
    def load_masks(self, idx):
        mod_idx = idx + (self.prev_len - 1)
        masks = self.mask_torch[mod_idx - self.input_dim + 1: mod_idx + self.output_dim + 1].type(torch.BoolTensor)
        return masks
    
    def __len__(self) -> int:
        return (len(self.times) - (self.prev_len - 1) - self.output_dim)
    
    def __getitem__(self, idx):

        mod_idx = idx + (self.prev_len - 1)

        feats = self.load_feats(idx)
        masks = self.load_masks(idx)

        raw_times = []

        cur_mask = masks[0].clone()

        pred_pm25_vals = self.feat_torch[mod_idx+1:mod_idx+1+self.output_dim, :self.korea_stn_num, 0].numpy() 

        pred_pm25_mask = self.feat_torch[mod_idx+1:mod_idx+1+self.output_dim, :self.korea_stn_num, 6].numpy().astype(bool)

        feat_dim = self.feat_torch.shape[-1]

        for t_idx in range(self.input_dim + self.output_dim):
            cur_idx = mod_idx - self.input_dim + 1 + t_idx
            raw_times.append([self.times[cur_idx].year, self.times[cur_idx].month, self.times[cur_idx].day, self.times[cur_idx].hour])

        simulation = torch.zeros(self.cmaq_size[0], self.cmaq_size[1], (self.input_dim + self.output_dim) * ((feat_dim // 2) * 4 + 4))
        reanalysis = torch.zeros(self.output_dim, self.cmaq_size[0], self.cmaq_size[1])
        curr_reanalysis = torch.zeros(self.cmaq_size[0], self.cmaq_size[1])

        cur_t_utc = self.times[mod_idx] - timedelta(hours=9)
        cur_f_name = f"{self.reanalysis_data_path}/{cur_t_utc.year}/ACONC.PM_RQ40i8a.KNU_09_01.{cur_t_utc.strftime('%Y%m%d')}.nc"
        ds = xr.open_dataset(cur_f_name)
        cur_pm25 = ds['PM2P5'].values
        cur_pm25 = cur_pm25[cur_t_utc.hour, 0]
        curr_reanalysis = torch.FloatTensor(cur_pm25)

        prev_pm25_vals = torch.zeros(self.prev_len, self.cmaq_size[0], self.cmaq_size[1])

        for t_idx in range(self.output_dim):
            cur_t_utc = self.times[mod_idx] - timedelta(hours=9) + timedelta(hours=t_idx + 1)
            cur_f_name = f"{self.reanalysis_data_path}/{cur_t_utc.year}/ACONC.PM_RQ40i8a.KNU_09_01.{cur_t_utc.strftime('%Y%m%d')}.nc"
            ds = xr.open_dataset(cur_f_name)
            cur_pm25 = ds['PM2P5'].values
            cur_pm25 = cur_pm25[cur_t_utc.hour, 0]
            reanalysis[t_idx] = torch.FloatTensor(cur_pm25)

        for t_idx in range(self.prev_len - self.input_dim):
            cur_idx = idx + t_idx
            cur_t_utc = self.times[cur_idx] - timedelta(hours=9)
            lead_from_03h = cur_t_utc.hour + 21
            if lead_from_03h >= 24:
                date_03h = cur_t_utc - timedelta(days=1)
            else:
                date_03h = cur_t_utc - timedelta(days=2)
                lead_from_03h += 24
            lead_from_09h = cur_t_utc.hour + 15
            if lead_from_09h >= 18:
                date_09h = cur_t_utc - timedelta(days=1)
            else:
                date_09h = cur_t_utc - timedelta(days=2)
                lead_from_09h += 24
            lead_from_15h = cur_t_utc.hour + 9
            if lead_from_15h >= 12:
                date_15h = cur_t_utc - timedelta(days=1)
            else:
                date_15h = cur_t_utc - timedelta(days=2)
                lead_from_15h += 24
            lead_from_21h = cur_t_utc.hour + 3
            if lead_from_21h >= 6:
                date_21h = cur_t_utc - timedelta(days=1)
            else:
                date_21h = cur_t_utc - timedelta(days=2)
                lead_from_21h += 24

            cur_sim_f_03h_name = f"{self.sim_data_path}/{date_03h.year}/" + date_03h.strftime("%m%d") + f"03_{lead_from_03h:02d}.npy"
            if not os.path.exists(cur_sim_f_03h_name):
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_03h = np.load(cur_sim_f_03h_name)
            if len(cur_sim_data_03h.shape) != 3:
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            cur_pm_25_03h = cur_sim_data_03h[4]

            cur_sim_f_09h_name = f"{self.sim_data_path}/{date_09h.year}/" + date_09h.strftime("%m%d") + f"09_{lead_from_09h:02d}.npy"
            if not os.path.exists(cur_sim_f_09h_name):
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_09h = np.load(cur_sim_f_09h_name)
            if len(cur_sim_data_09h.shape) != 3:
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            cur_pm_25_09h = cur_sim_data_09h[4]

            cur_sim_f_15h_name = f"{self.sim_data_path}/{date_15h.year}/" + date_15h.strftime("%m%d") + f"15_{lead_from_15h:02d}.npy"
            if not os.path.exists(cur_sim_f_15h_name):
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_15h = np.load(cur_sim_f_15h_name)
            if len(cur_sim_data_15h.shape) != 3:
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            cur_pm_25_15h = cur_sim_data_15h[4]
            
            cur_sim_f_21h_name = f"{self.sim_data_path}/{date_21h.year}/" + date_21h.strftime("%m%d") + f"21_{lead_from_21h:02d}.npy"
            if not os.path.exists(cur_sim_f_21h_name):
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_21h = np.load(cur_sim_f_21h_name)
            if len(cur_sim_data_21h.shape) != 3:
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            cur_pm_25_21h = cur_sim_data_21h[4]

            prev_pm25_vals[t_idx, :, :] = torch.FloatTensor(np.mean([cur_pm_25_03h, cur_pm_25_09h, cur_pm_25_15h, cur_pm_25_21h], axis=0))
        
        for t_idx in range(self.input_dim):
            cur_idx = mod_idx - self.input_dim + 1 + t_idx
            cur_t_utc = self.times[cur_idx] - timedelta(hours=9)
            lead_from_03h = cur_t_utc.hour + 21
            if lead_from_03h >= 24:
                date_03h = cur_t_utc - timedelta(days=1)
            else:
                date_03h = cur_t_utc - timedelta(days=2)
                lead_from_03h += 24
            lead_from_09h = cur_t_utc.hour + 15
            if lead_from_09h >= 18:
                date_09h = cur_t_utc - timedelta(days=1)
            else:
                date_09h = cur_t_utc - timedelta(days=2)
                lead_from_09h += 24
            lead_from_15h = cur_t_utc.hour + 9
            if lead_from_15h >= 12:
                date_15h = cur_t_utc - timedelta(days=1)
            else:
                date_15h = cur_t_utc - timedelta(days=2)
                lead_from_15h += 24
            lead_from_21h = cur_t_utc.hour + 3
            if lead_from_21h >= 6:
                date_21h = cur_t_utc - timedelta(days=1)
            else:
                date_21h = cur_t_utc - timedelta(days=2)
                lead_from_21h += 24
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4] = lead_from_03h
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 1] = lead_from_09h
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 2] = lead_from_15h
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 3] = lead_from_21h

            cur_sim_f_03h_name = f"{self.sim_data_path}/{date_03h.year}/" + date_03h.strftime("%m%d") + f"03_{lead_from_03h:02d}.npy"
            if not os.path.exists(cur_sim_f_03h_name):
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_03h = np.load(cur_sim_f_03h_name)
            if len(cur_sim_data_03h.shape) != 3:
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))

            cur_sim_data_03h[0] = (cur_sim_data_03h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_03h[1] = (cur_sim_data_03h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_03h[2] = (cur_sim_data_03h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_03h[3] = (cur_sim_data_03h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_03h[5] = (cur_sim_data_03h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_pm_25_03h = cur_sim_data_03h[4]

            cur_sim_data_03h = np.moveaxis(cur_sim_data_03h, 0, -1)
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4): t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2)] = torch.FloatTensor(cur_sim_data_03h)

            cur_sim_f_09h_name = f"{self.sim_data_path}/{date_09h.year}/" + date_09h.strftime("%m%d") + f"09_{lead_from_09h:02d}.npy"
            if not os.path.exists(cur_sim_f_09h_name):
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_09h = np.load(cur_sim_f_09h_name)
            if len(cur_sim_data_09h.shape) != 3:
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))

            cur_sim_data_09h[0] = (cur_sim_data_09h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_09h[1] = (cur_sim_data_09h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_09h[2] = (cur_sim_data_09h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_09h[3] = (cur_sim_data_09h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_09h[5] = (cur_sim_data_09h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_pm_25_09h = cur_sim_data_09h[4]

            cur_sim_data_09h = np.moveaxis(cur_sim_data_09h, 0, -1)
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2): t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 2] = torch.FloatTensor(cur_sim_data_09h)

            cur_sim_f_15h_name = f"{self.sim_data_path}/{date_15h.year}/" + date_15h.strftime("%m%d") + f"15_{lead_from_15h:02d}.npy"
            if not os.path.exists(cur_sim_f_15h_name):
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_15h = np.load(cur_sim_f_15h_name)
            if len(cur_sim_data_15h.shape) != 3:
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            
            cur_sim_data_15h[0] = (cur_sim_data_15h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_15h[1] = (cur_sim_data_15h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_15h[2] = (cur_sim_data_15h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_15h[3] = (cur_sim_data_15h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_15h[5] = (cur_sim_data_15h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_pm_25_15h = cur_sim_data_15h[4]

            cur_sim_data_15h = np.moveaxis(cur_sim_data_15h, 0, -1)
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 2: t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 3] = torch.FloatTensor(cur_sim_data_15h)

            cur_sim_f_21h_name = f"{self.sim_data_path}/{date_21h.year}/" + date_21h.strftime("%m%d") + f"21_{lead_from_21h:02d}.npy"
            if not os.path.exists(cur_sim_f_21h_name):
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_21h = np.load(cur_sim_f_21h_name)
            if len(cur_sim_data_21h.shape) != 3:
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            
            cur_sim_data_21h[0] = (cur_sim_data_21h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_21h[1] = (cur_sim_data_21h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_21h[2] = (cur_sim_data_21h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_21h[3] = (cur_sim_data_21h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_21h[5] = (cur_sim_data_21h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_pm_25_21h = cur_sim_data_21h[4]

            cur_sim_data_21h = np.moveaxis(cur_sim_data_21h, 0, -1)
            simulation[:,:, t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 3: t_idx * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4] = torch.FloatTensor(cur_sim_data_21h)

            prev_pm25_vals[t_idx + (self.prev_len - self.input_dim), :, :] = torch.FloatTensor(np.mean([cur_pm_25_03h, cur_pm_25_09h, cur_pm_25_15h, cur_pm_25_21h], axis=0))
        

        for t_idx in range(self.output_dim):

            cur_idx = mod_idx + t_idx + 1
            cur_t_utc = self.times[cur_idx] - timedelta(hours=9)
            lead_from_03h = cur_t_utc.hour + 21
            if lead_from_03h >= 24:
                date_03h = cur_t_utc - timedelta(days=1)
            else:
                date_03h = cur_t_utc - timedelta(days=2)
                lead_from_03h += 24
            lead_from_09h = cur_t_utc.hour + 15
            if lead_from_09h >= 18:
                date_09h = cur_t_utc - timedelta(days=1)
            else:
                date_09h = cur_t_utc - timedelta(days=2)
                lead_from_09h += 24
            lead_from_15h = cur_t_utc.hour + 9
            if lead_from_15h >= 12:
                date_15h = cur_t_utc - timedelta(days=1)
            else:
                date_15h = cur_t_utc - timedelta(days=2)
                lead_from_15h += 24
            lead_from_21h = cur_t_utc.hour + 3
            if lead_from_21h >= 6:
                date_21h = cur_t_utc - timedelta(days=1)
            else:
                date_21h = cur_t_utc - timedelta(days=2)
                lead_from_21h += 24
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4] = lead_from_03h
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 1] = lead_from_09h
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 2] = lead_from_15h
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4 + 3] = lead_from_21h

            cur_sim_f_03h_name = f"{self.sim_data_path}/{date_03h.year}/" + date_03h.strftime("%m%d") + f"03_{lead_from_03h:02d}.npy"
            if not os.path.exists(cur_sim_f_03h_name):
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_03h = np.load(cur_sim_f_03h_name)
            if len(cur_sim_data_03h.shape) != 3:
                cur_sim_data_03h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))

            cur_sim_data_03h[0] = (cur_sim_data_03h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_03h[1] = (cur_sim_data_03h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_03h[2] = (cur_sim_data_03h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_03h[3] = (cur_sim_data_03h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_03h[5] = (cur_sim_data_03h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]

            cur_sim_data_03h = np.moveaxis(cur_sim_data_03h, 0, -1)
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4): (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2)] = torch.FloatTensor(cur_sim_data_03h)

            cur_sim_f_09h_name = f"{self.sim_data_path}/{date_09h.year}/" + date_09h.strftime("%m%d") + f"09_{lead_from_09h:02d}.npy"
            if not os.path.exists(cur_sim_f_09h_name):
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_09h = np.load(cur_sim_f_09h_name)
            if len(cur_sim_data_09h.shape) != 3:
                cur_sim_data_09h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))

            cur_sim_data_09h[0] = (cur_sim_data_09h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_09h[1] = (cur_sim_data_09h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_09h[2] = (cur_sim_data_09h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_09h[3] = (cur_sim_data_09h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_09h[5] = (cur_sim_data_09h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]

            cur_sim_data_09h = np.moveaxis(cur_sim_data_09h, 0, -1)
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2): (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 2] = torch.FloatTensor(cur_sim_data_09h)

            cur_sim_f_15h_name = f"{self.sim_data_path}/{date_15h.year}/" + date_15h.strftime("%m%d") + f"15_{lead_from_15h:02d}.npy"
            if not os.path.exists(cur_sim_f_15h_name):
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_15h = np.load(cur_sim_f_15h_name)
            if len(cur_sim_data_15h.shape) != 3:
                cur_sim_data_15h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            
            cur_sim_data_15h[0] = (cur_sim_data_15h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_15h[1] = (cur_sim_data_15h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_15h[2] = (cur_sim_data_15h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_15h[3] = (cur_sim_data_15h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_15h[5] = (cur_sim_data_15h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]

            cur_sim_data_15h = np.moveaxis(cur_sim_data_15h, 0, -1)
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 2: (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 3] = torch.FloatTensor(cur_sim_data_15h)

            cur_sim_f_21h_name = f"{self.sim_data_path}/{date_21h.year}/" + date_21h.strftime("%m%d") + f"21_{lead_from_21h:02d}.npy"
            if not os.path.exists(cur_sim_f_21h_name):
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            else:
                cur_sim_data_21h = np.load(cur_sim_f_21h_name)
            if len(cur_sim_data_21h.shape) != 3:
                cur_sim_data_21h = np.zeros((feat_dim // 2, self.cmaq_size[0], self.cmaq_size[1]))
            
            cur_sim_data_21h[0] = (cur_sim_data_21h[0] - self.feat_infos['CO'][0]) / self.feat_infos['CO'][1]
            cur_sim_data_21h[1] = (cur_sim_data_21h[1] - self.feat_infos['NO2'][0]) / self.feat_infos['NO2'][1]
            cur_sim_data_21h[2] = (cur_sim_data_21h[2] - self.feat_infos['O3'][0]) / self.feat_infos['O3'][1]
            cur_sim_data_21h[3] = (cur_sim_data_21h[3] - self.feat_infos['PM10'][0]) / self.feat_infos['PM10'][1]
            cur_sim_data_21h[5] = (cur_sim_data_21h[5] - self.feat_infos['SO2'][0]) / self.feat_infos['SO2'][1]
            cur_sim_data_21h = np.moveaxis(cur_sim_data_21h, 0, -1)
            simulation[:,:, (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 3: (t_idx + self.input_dim) * ((feat_dim // 2) * 4 + 4) + (feat_dim // 2) * 4] = torch.FloatTensor(cur_sim_data_21h)


        range_4class = [(-1,15),(15,35),(35,75), (75,np.Inf)]
        class_four = [0,1,2,3]
        reanalysis_class = assign_class(reanalysis.numpy(), range_4class, class_four)

        pred_pm25_class = assign_class2(pred_pm25_vals, pred_pm25_mask, range_4class, class_four)

        return feats, masks, simulation, curr_reanalysis, reanalysis, torch.IntTensor(reanalysis_class), torch.FloatTensor(raw_times), prev_pm25_vals, torch.FloatTensor(pred_pm25_vals), torch.BoolTensor(pred_pm25_mask), torch.IntTensor(pred_pm25_class)

    def collate_fn(self, samples):
        feats, masks, simulation, curr_reanalysis, reanalysis, reanalysis_class, raw_times, prev_vals, pred_vals, pred_mask, pred_classes = zip(*samples)
        feats = torch.stack(feats, dim=0)
        masks = torch.stack(masks, dim=0)
        pred_classes = torch.stack(pred_classes, dim=0)
        pred_vals = torch.stack(pred_vals, dim=0)
        pred_mask = torch.stack(pred_mask, dim=0)
        simulation = torch.stack(simulation, dim=0)
        curr_reanalysis = torch.stack(curr_reanalysis, dim=0)
        reanalysis = torch.stack(reanalysis, dim=0)
        reanalysis_class = torch.stack(reanalysis_class, dim=0)
        raw_times = torch.stack(raw_times)
        prev_vals = torch.stack(prev_vals, dim=0)
        return feats, masks, simulation, curr_reanalysis, reanalysis, reanalysis_class, raw_times, prev_vals, pred_vals, pred_mask, pred_classes
    





