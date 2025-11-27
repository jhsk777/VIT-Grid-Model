import os
# OMP_NUM_THREADS: openmp, OPENBLAS_NUM_THREADS: openblas, MKL_NUM_THREADS: mkl, VECLIB_MAXIMUM_THREADS: accelerate, NUMEXPR_NUM_THREADS: numexpr
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
# os.environ["OPENBLAS_NUM_THREADS"] = "2" # export OPENBLAS_NUM_THREADS=4
# os.environ["VECLIB_MAXIMUM_THREADS"] = "2" # export VECLIB_MAXIMUM_THREADS=4

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import Air_Simulation_Reanalysis_Dataset_v3, Air_Simulation_Reanalysis_Dataset_only
# from model import simulation_grid_model
from model import simulation_grid_model_v3
from metnet3 import MetNet3

import random

import argparse

from datetime import datetime, timedelta
import pandas as pd

import pdb
import xarray as xr


# assign PM2.5 class accorrding to the range
def assign_class(arr, range, classes):
  return np.select([np.logical_and(arr > r[0], arr <= r[1]) for r in range], classes, default=0)

# load station coordinates
def load_stations(args):
    lats = []
    lons = []
    korea_stn_regions = []
    korea_stn_num = 0
    china_stn_num = 0
    with open(f"{args.data_path}/station_infos/korea.txt", 'r') as f:
        for line in f:
            data = line.strip().split(",")
            lats.append(float(data[2]))
            lons.append(float(data[3]))
            korea_stn_regions.append(data[-1])
            korea_stn_num += 1
    with open(f"{args.data_path}/station_infos/china.txt", 'r') as f:
        for line in f:
            data = line.strip().split(",")
            lats.append(float(data[2]))
            lons.append(float(data[3]))
            china_stn_num += 1
    lats = torch.tensor(lats)
    lons = torch.tensor(lons)
    return lats, lons, korea_stn_regions, korea_stn_num, china_stn_num

# Total evaluation code
def evaluation(args):

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    feat_dim = args.feat_dim
    input_dim = args.input_dim
    output_dim = args.output_dim

    lats, lons, korea_stn_regions, korea_stn_num, china_stn_num = load_stations(args)

    region_names = list(set(korea_stn_regions))
    stn_to_region_idx = []

    num_rgn = len(region_names)

    for rgn in korea_stn_regions:
        stn_to_region_idx.append(region_names.index(rgn))

    stn_to_region_idx = torch.IntTensor(stn_to_region_idx)

    sim_coords = np.zeros((korea_stn_num, 2), dtype=int)

    with open(f"{args.data_path}/station_infos/coords.txt", 'r') as f:
        for i, line in enumerate(f):
            data = line.strip().split(",")
            sim_coords[i] = [int(data[0]), int(data[1])]

    cmaq_size = (82, 67)
    cmaq_coords = np.zeros((cmaq_size[0], cmaq_size[1], 2), dtype=float)

    ds = xr.open_dataset(f"{args.data_path}/station_infos/GRID_INFO_09km.nc")
    cmaq_coords[:,:,0] = ds['LAT'].values
    cmaq_coords[:,:,1] = ds['LON'].values

    feat_infos = {}
    with open(f"{args.data_path}/feat_infos.txt", "r") as f:
        for line in f.readlines():
            feat_name, mean, std = line.strip().split(',')
            if feat_name == 'feature':
                continue
            feat_infos[feat_name] = (float(mean), float(std))

    # load model
    sample_size = (args.input_dim+args.output_dim, 24, cmaq_size[0], cmaq_size[1])
    vit_model = MetNet3(input_size_sample=sample_size, n_start_channels=args.hidden_dim, end_lead_time=args.output_dim, pm25_boundaries=[15,35,75], pm10_boundaries=[15,35,75], pm25_mean=feat_infos['PM2.5'][0], pm25_std=feat_infos['PM2.5'][1])
    vit_model = torch.nn.DataParallel(vit_model, device_ids=args.all_devices, output_device=args.default_device)

    vit_model.load_state_dict(torch.load(f'check_points/{args.model_name}.pkt', map_location=f'cuda:{args.default_device}'))
    
    test_start = datetime(2023, 1, 1, 0)
    test_end = datetime(2023, 3, 31, 23)

    test_times = []

    cur_t = test_start - timedelta(hours=(args.prev_len - 1))

    while cur_t <= (test_end + timedelta(hours=output_dim)):
        test_times.append(cur_t)
        cur_t += timedelta(hours=1)
    
    test_time_len = len(test_times)

    feat_test = np.zeros((test_time_len, korea_stn_num + china_stn_num, feat_dim))
    mask_test = np.zeros((test_time_len, korea_stn_num + china_stn_num))

    idx = 0
    for t in tqdm(test_times, desc='loading test data'):
        cur_f_name = f"{args.data_path}/ground_obs/{t.year}/{t.month}/" + t.strftime("%d%H") + ".npy"
        cur_data = np.load(cur_f_name)
        feat_test[idx] = cur_data[:, :feat_dim]
        mask_test[idx] = cur_data[:, -1]
        idx += 1
    
    test_time_len = len(test_times)

    test_dataset = Air_Simulation_Reanalysis_Dataset_only(test_times, feat_test, mask_test, input_dim, output_dim, args.prev_len, korea_stn_num, china_stn_num, cmaq_size, args.sim_data_path, args.analysis_data_path, feat_infos)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=5, collate_fn = test_dataset.collate_fn, drop_last=False)

    criterion = torch.nn.MSELoss()
        
    val_loss_sum, val_acc = 0., 0
    _TP1, _TN1, _FP1, _FN1 = 0, 0, 0, 0
    _TP2, _TN2, _FP2, _FN2 = 0, 0, 0, 0
    _TP3, _TN3, _FP3, _FN3 = 0, 0, 0, 0
    total_a1, total_a2, total_a3, total_a4 = 0, 0, 0, 0
    total_b1, total_b2, total_b3, total_b4 = 0, 0, 0, 0
    total_c1, total_c2, total_c3, total_c4 = 0, 0, 0, 0
    total_d1, total_d2, total_d3, total_d4 = 0, 0, 0, 0

    per_total_a1, per_total_a2, per_total_a3, per_total_a4 = 0, 0, 0, 0
    per_total_b1, per_total_b2, per_total_b3, per_total_b4 = 0, 0, 0, 0
    per_total_c1, per_total_c2, per_total_c3, per_total_c4 = 0, 0, 0, 0
    per_total_d1, per_total_d2, per_total_d3, per_total_d4 = 0, 0, 0, 0

    sim_21h_total_a1, sim_21h_total_a2, sim_21h_total_a3, sim_21h_total_a4 = 0, 0, 0, 0
    sim_21h_total_b1, sim_21h_total_b2, sim_21h_total_b3, sim_21h_total_b4 = 0, 0, 0, 0
    sim_21h_total_c1, sim_21h_total_c2, sim_21h_total_c3, sim_21h_total_c4 = 0, 0, 0, 0
    sim_21h_total_d1, sim_21h_total_d2, sim_21h_total_d3, sim_21h_total_d4 = 0, 0, 0, 0

    sim_avg_total_a1, sim_avg_total_a2, sim_avg_total_a3, sim_avg_total_a4 = 0, 0, 0, 0
    sim_avg_total_b1, sim_avg_total_b2, sim_avg_total_b3, sim_avg_total_b4 = 0, 0, 0, 0
    sim_avg_total_c1, sim_avg_total_c2, sim_avg_total_c3, sim_avg_total_c4 = 0, 0, 0, 0
    sim_avg_total_d1, sim_avg_total_d2, sim_avg_total_d3, sim_avg_total_d4 = 0, 0, 0, 0

    _TP = np.zeros(3 * output_dim)
    _TN = np.zeros(3 * output_dim)
    _FP = np.zeros(3 * output_dim)
    _FN = np.zeros(3 * output_dim)
    p_TP = np.zeros(3 * output_dim)
    p_TN = np.zeros(3 * output_dim)
    p_FP = np.zeros(3 * output_dim)
    p_FN = np.zeros(3 * output_dim)
    
    sim_21h_TP = np.zeros(3 * output_dim)
    sim_21h_TN = np.zeros(3 * output_dim)
    sim_21h_FP = np.zeros(3 * output_dim)
    sim_21h_FN = np.zeros(3 * output_dim)
    sim_avg_TP = np.zeros(3 * output_dim)
    sim_avg_TN = np.zeros(3 * output_dim)
    sim_avg_FP = np.zeros(3 * output_dim)
    sim_avg_FN = np.zeros(3 * output_dim)

    _RMSE_np = np.zeros(3 * output_dim)
    _MAE_np = np.zeros(3 * output_dim)
    p_RMSE_np = np.zeros(3 * output_dim)
    p_MAE_np = np.zeros(3 * output_dim)
    sim_21h_RMSE_np = np.zeros(3 * output_dim)
    sim_21h_MAE_np = np.zeros(3 * output_dim)
    sim_avg_RMSE_np = np.zeros(3 * output_dim)
    sim_avg_MAE_np = np.zeros(3 * output_dim)
    valid_count = np.zeros(3 * output_dim)

    range_4class = [(-1,15),(15,35),(35,75), (75,np.Inf)]
    class_four = [0,1,2,3]

    gt_vals = []
    gt_masks = []
    model_vals = []
    multiair_vals = []
    times = []

    f_log = open("logs/test_"+args.model_name+".log", "a")
    f_log.write(str(args))
    f_log.write('\n')
    f_log.flush()

    
    # model evaluation start
    vit_model.eval()
    with torch.no_grad():
        val_step_cnt = 0
        valid_diff_sum = 0
        valid_diff_sum_p = 0
        valid_diff_sum_sim_21h = 0
        valid_diff_sum_sim_avg = 0
        valid_diff_squares_sum = 0
        valid_diff_squares_sum_p = 0
        valid_diff_squares_sum_sim_21h = 0
        valid_diff_squares_sum_sim_avg = 0
        valid_entry_cnt = 0

        valid_norm_diff_sum = 0
        valid_norm_diff_sum_p = 0
        valid_norm_diff_sum_sim_21h = 0
        valid_norm_diff_sum_sim_avg = 0
        valid_norm_diff_abs_sum = 0
        valid_norm_diff_abs_sum_p = 0
        valid_norm_diff_abs_sum_sim_21h = 0
        valid_norm_diff_abs_sum_sim_avg = 0

        valid_vals_gt = []
        valid_vals_model = []
        valid_vals_p = []
        valid_vals_sim_21h = []
        valid_vals_sim_avg = []
        valid_nonzero_entry_cnt = 0

        for simulation, curr_reanalysis, reanalysis, reanalysis_classes, raw_times, prev_vals in tqdm(iter(test_loader), desc='evaluation'):
            # pdb.set_trace()
            last_PM = curr_reanalysis
            last_PM = last_PM.reshape(simulation.shape[0], 1, cmaq_size[0] * cmaq_size[1])
            last_PM = last_PM.repeat(1, output_dim, 1).view(last_PM.shape[0], output_dim, cmaq_size[0] * cmaq_size[1]).to(args.default_device)
            last_PM_np = np.array(last_PM.cpu().numpy())
            persistent_PM = assign_class(last_PM_np, range_4class, class_four) #
            persistent_PM = torch.from_numpy(persistent_PM).to(args.default_device)
            simulation = simulation.to(args.default_device)
            simulation_for_vit = simulation.reshape(simulation.shape[0], cmaq_size[0], cmaq_size[1], args.input_dim+args.output_dim, -1).permute(0,3,4,1,2)
            simulation_for_vit = simulation_for_vit[:,:,:-4,:,:]
            preds = vit_model(simulation_for_vit, timestamps=raw_times)
            # preds = model(feats, masks, raw_times[:,:,1:], prev_vals, simulation)
            preds = preds.reshape(simulation.shape[0], output_dim, cmaq_size[0] * cmaq_size[1])
            # preds = preds.permute(0, 2, 1)
            preds[preds < 0.] = 0.

            if torch.sum(torch.isnan(preds)) > 0:
                pdb.set_trace()
            
            predict_PM_np = np.array(preds.detach().cpu().numpy())
            predict_PM_class = assign_class(predict_PM_np, range_4class, class_four)
            labels = torch.from_numpy(predict_PM_class).to(args.default_device)
            pred_classes = reanalysis_classes.to(args.default_device)
            pred_classes = pred_classes.reshape(preds.shape[0], output_dim, cmaq_size[0] * cmaq_size[1])
            
            pred_vals = reanalysis.reshape(simulation.shape[0], output_dim, cmaq_size[0] * cmaq_size[1]).to(args.default_device)

            sim_21h_pm_vals = torch.zeros_like(preds).to(args.default_device)
            sim_avg_pm_vals = torch.zeros_like(preds).to(args.default_device)


            for i in range(output_dim):
                # cur_vals = simulation_pm
                # sim_pm_vals[:,i] = cur_vals
                cur_vals = simulation[:,:,:,(i + input_dim) * ((feat_dim // 2) * 4 + 4):(i + input_dim + 1) * ((feat_dim // 2) * 4 + 4)]
                sim_21h_pm_vals[:, i] = cur_vals[:,:,:,22].reshape(cur_vals.shape[0], cmaq_size[0] * cmaq_size[1])
                sim_avg_pm_vals[:, i] = torch.mean(cur_vals[:,:,:,[4, 10, 16, 22]], dim=3).reshape(cur_vals.shape[0], cmaq_size[0] * cmaq_size[1])
            # pdb.set_trace()
            sim_21h_pm_vals_np = np.array(sim_21h_pm_vals.cpu().numpy())
            sim_21h_pm_class = assign_class(sim_21h_pm_vals_np, range_4class, class_four)
            sim_21h_labels = torch.from_numpy(sim_21h_pm_class).to(args.default_device)
            sim_avg_pm_vals_np = np.array(sim_avg_pm_vals.cpu().numpy())
            sim_avg_pm_class = assign_class(sim_avg_pm_vals_np, range_4class, class_four)
            sim_avg_labels = torch.from_numpy(sim_avg_pm_class).to(args.default_device)

            valid_time_mask = raw_times[:,input_dim - 1,3] == 6.
            valid_times = raw_times[valid_time_mask, input_dim - 1]
            valid_times = valid_times[:,0].type(torch.IntTensor) * 1000000 + valid_times[:,1].type(torch.IntTensor) * 10000 + valid_times[:,2].type(torch.IntTensor) * 100 + valid_times[:,3].type(torch.IntTensor) 
        
            times.append(valid_times)

            val_loss_sum += criterion(preds, pred_vals).item()

            valid_diffs = preds - pred_vals
            valid_diff_sum += torch.abs(valid_diffs).sum().item()
            valid_diff_squares_sum += (valid_diffs ** 2).sum().item()
            
            valid_diffs_p = last_PM - pred_vals
            valid_diff_sum_p += torch.abs(valid_diffs_p).sum().item()
            valid_diff_squares_sum_p += (valid_diffs_p ** 2).sum().item()

            valid_diffs_sim_21h = sim_21h_pm_vals - pred_vals
            valid_diff_sum_sim_21h += torch.abs(valid_diffs_sim_21h).sum().item()
            valid_diff_squares_sum_sim_21h += (valid_diffs_sim_21h ** 2).sum().item()

            valid_diffs_sim_avg = sim_avg_pm_vals - pred_vals
            valid_diff_sum_sim_avg += torch.abs(valid_diffs_sim_avg).sum().item()
            valid_diff_squares_sum_sim_avg += (valid_diffs_sim_avg ** 2).sum().item()

            valid_entry_cnt += torch.numel(preds)

            nonzero_mask = (pred_vals > 0)
            valid_nonzero_entry_cnt += nonzero_mask.sum().item()

            valid_nonzero_diffs = preds[nonzero_mask] - pred_vals[nonzero_mask]
            valid_norm_diff_sum += ((valid_nonzero_diffs) / pred_vals[nonzero_mask]).sum().item()
            valid_norm_diff_abs_sum += torch.abs((valid_nonzero_diffs) / pred_vals[nonzero_mask]).sum().item()

            valid_nonzero_diffs_p = last_PM[nonzero_mask] - pred_vals[nonzero_mask]
            valid_norm_diff_sum_p += ((valid_nonzero_diffs_p) / pred_vals[nonzero_mask]).sum().item()
            valid_norm_diff_abs_sum_p += torch.abs((valid_nonzero_diffs_p) / pred_vals[nonzero_mask]).sum().item()

            valid_nonzero_diffs_sim_21h = sim_21h_pm_vals[nonzero_mask] - pred_vals[nonzero_mask]
            valid_norm_diff_sum_sim_21h += ((valid_nonzero_diffs_sim_21h) / pred_vals[nonzero_mask]).sum().item()
            valid_norm_diff_abs_sum_sim_21h += torch.abs((valid_nonzero_diffs_sim_21h) / pred_vals[nonzero_mask]).sum().item()

            valid_vals_gt += pred_vals.tolist()
            valid_vals_model += preds.tolist()
            valid_vals_p += last_PM.tolist()
            valid_vals_sim_21h += sim_21h_pm_vals.tolist()
            valid_vals_sim_avg += sim_avg_pm_vals.tolist()

            cur_labels = labels
            cur_preds = pred_classes
            cur_perst = persistent_PM
            cur_sim_21h = sim_21h_labels
            cur_sim_avg = sim_avg_labels

            val_acc += ((cur_labels == cur_preds)).float().sum().item()
            
            val_step_cnt += 1

            # count of occurrences for all combinations of ground and predicted classes
            total_a1 += ((cur_labels == 0) & (cur_preds == 0)).float().sum().item()
            total_a2 += ((cur_labels == 0) & (cur_preds == 1)).float().sum().item()
            total_a3 += ((cur_labels == 0) & (cur_preds == 2)).float().sum().item()
            total_a4 += ((cur_labels == 0) & (cur_preds == 3)).float().sum().item()
            total_b1 += ((cur_labels == 1) & (cur_preds == 0)).float().sum().item()
            total_b2 += ((cur_labels == 1) & (cur_preds == 1)).float().sum().item()
            total_b3 += ((cur_labels == 1) & (cur_preds == 2)).float().sum().item()
            total_b4 += ((cur_labels == 1) & (cur_preds == 3)).float().sum().item()
            total_c1 += ((cur_labels == 2) & (cur_preds == 0)).float().sum().item()
            total_c2 += ((cur_labels == 2) & (cur_preds == 1)).float().sum().item()
            total_c3 += ((cur_labels == 2) & (cur_preds == 2)).float().sum().item()
            total_c4 += ((cur_labels == 2) & (cur_preds == 3)).float().sum().item()
            total_d1 += ((cur_labels == 3) & (cur_preds == 0)).float().sum().item()
            total_d2 += ((cur_labels == 3) & (cur_preds == 1)).float().sum().item()
            total_d3 += ((cur_labels == 3) & (cur_preds == 2)).float().sum().item()
            total_d4 += ((cur_labels == 3) & (cur_preds == 3)).float().sum().item()


            per_total_a1 += ((cur_perst == 0) & (cur_preds == 0)).float().sum().item()
            per_total_a2 += ((cur_perst == 0) & (cur_preds == 1)).float().sum().item()
            per_total_a3 += ((cur_perst == 0) & (cur_preds == 2)).float().sum().item()
            per_total_a4 += ((cur_perst == 0) & (cur_preds == 3)).float().sum().item()
            per_total_b1 += ((cur_perst == 1) & (cur_preds == 0)).float().sum().item()
            per_total_b2 += ((cur_perst == 1) & (cur_preds == 1)).float().sum().item()
            per_total_b3 += ((cur_perst == 1) & (cur_preds == 2)).float().sum().item()
            per_total_b4 += ((cur_perst == 1) & (cur_preds == 3)).float().sum().item()
            per_total_c1 += ((cur_perst == 2) & (cur_preds == 0)).float().sum().item()
            per_total_c2 += ((cur_perst == 2) & (cur_preds == 1)).float().sum().item()
            per_total_c3 += ((cur_perst == 2) & (cur_preds == 2)).float().sum().item()
            per_total_c4 += ((cur_perst == 2) & (cur_preds == 3)).float().sum().item()
            per_total_d1 += ((cur_perst == 3) & (cur_preds == 0)).float().sum().item()
            per_total_d2 += ((cur_perst == 3) & (cur_preds == 1)).float().sum().item()
            per_total_d3 += ((cur_perst == 3) & (cur_preds == 2)).float().sum().item()
            per_total_d4 += ((cur_perst == 3) & (cur_preds == 3)).float().sum().item()

            sim_21h_total_a1 += ((cur_sim_21h == 0) & (cur_preds == 0)).float().sum().item()
            sim_21h_total_a2 += ((cur_sim_21h == 0) & (cur_preds == 1)).float().sum().item()
            sim_21h_total_a3 += ((cur_sim_21h == 0) & (cur_preds == 2)).float().sum().item()
            sim_21h_total_a4 += ((cur_sim_21h == 0) & (cur_preds == 3)).float().sum().item()
            sim_21h_total_b1 += ((cur_sim_21h == 1) & (cur_preds == 0)).float().sum().item()
            sim_21h_total_b2 += ((cur_sim_21h == 1) & (cur_preds == 1)).float().sum().item()
            sim_21h_total_b3 += ((cur_sim_21h == 1) & (cur_preds == 2)).float().sum().item()
            sim_21h_total_b4 += ((cur_sim_21h == 1) & (cur_preds == 3)).float().sum().item()
            sim_21h_total_c1 += ((cur_sim_21h == 2) & (cur_preds == 0)).float().sum().item()
            sim_21h_total_c2 += ((cur_sim_21h == 2) & (cur_preds == 1)).float().sum().item()
            sim_21h_total_c3 += ((cur_sim_21h == 2) & (cur_preds == 2)).float().sum().item()
            sim_21h_total_c4 += ((cur_sim_21h == 2) & (cur_preds == 3)).float().sum().item()
            sim_21h_total_d1 += ((cur_sim_21h == 3) & (cur_preds == 0)).float().sum().item()
            sim_21h_total_d2 += ((cur_sim_21h == 3) & (cur_preds == 1)).float().sum().item()
            sim_21h_total_d3 += ((cur_sim_21h == 3) & (cur_preds == 2)).float().sum().item()
            sim_21h_total_d4 += ((cur_sim_21h == 3) & (cur_preds == 3)).float().sum().item()

            sim_avg_total_a1 += ((cur_sim_avg == 0) & (cur_preds == 0)).float().sum().item()
            sim_avg_total_a2 += ((cur_sim_avg == 0) & (cur_preds == 1)).float().sum().item()
            sim_avg_total_a3 += ((cur_sim_avg == 0) & (cur_preds == 2)).float().sum().item()
            sim_avg_total_a4 += ((cur_sim_avg == 0) & (cur_preds == 3)).float().sum().item()
            sim_avg_total_b1 += ((cur_sim_avg == 1) & (cur_preds == 0)).float().sum().item()
            sim_avg_total_b2 += ((cur_sim_avg == 1) & (cur_preds == 1)).float().sum().item()
            sim_avg_total_b3 += ((cur_sim_avg == 1) & (cur_preds == 2)).float().sum().item()
            sim_avg_total_b4 += ((cur_sim_avg == 1) & (cur_preds == 3)).float().sum().item()
            sim_avg_total_c1 += ((cur_sim_avg == 2) & (cur_preds == 0)).float().sum().item()
            sim_avg_total_c2 += ((cur_sim_avg == 2) & (cur_preds == 1)).float().sum().item()
            sim_avg_total_c3 += ((cur_sim_avg == 2) & (cur_preds == 2)).float().sum().item()
            sim_avg_total_c4 += ((cur_sim_avg == 2) & (cur_preds == 3)).float().sum().item()
            sim_avg_total_d1 += ((cur_sim_avg == 3) & (cur_preds == 0)).float().sum().item()
            sim_avg_total_d2 += ((cur_sim_avg == 3) & (cur_preds == 1)).float().sum().item()
            sim_avg_total_d3 += ((cur_sim_avg == 3) & (cur_preds == 2)).float().sum().item()
            sim_avg_total_d4 += ((cur_sim_avg == 3) & (cur_preds == 3)).float().sum().item()

            
            _TP1 += ((cur_labels > 0) & (cur_preds > 0)).float().sum().item()
            _TN1 += ((cur_labels == 0) & (cur_preds == 0)).float().sum().item()
            _FP1 += ((cur_labels > 0) & (cur_preds == 0)).float().sum().item()
            _FN1 += ((cur_labels == 0) & (cur_preds > 0)).float().sum().item()
            _TP2 += ((cur_labels > 1) & (cur_preds > 1)).float().sum().item()
            _TN2 += ((cur_labels < 2) & (cur_preds < 2)).float().sum().item()
            _FP2 += ((cur_labels > 1) & (cur_preds < 2)).float().sum().item()
            _FN2 += ((cur_labels < 2) & (cur_preds > 1)).float().sum().item()
            _TP3 += ((cur_labels > 2) & (cur_preds > 2)).float().sum().item()
            _TN3 += ((cur_labels < 3) & (cur_preds < 3)).float().sum().item()
            _FP3 += ((cur_labels > 2) & (cur_preds < 3)).float().sum().item()
            _FN3 += ((cur_labels < 3) & (cur_preds > 2)).float().sum().item()
            
            
            for i in range(1, 3+1):
                for j in range(output_dim):
                    cur_labels = labels[:,j]
                    cur_preds = pred_classes[:,j]
                    cur_perst = persistent_PM[:,j]
                    cur_sim_21h = sim_21h_labels[:,j]
                    cur_sim_avg = sim_avg_labels[:,j]

                    _TP[(i-1) * output_dim + j] += ((cur_labels > i-1) & (cur_preds > i-1)).sum()
                    _TN[(i-1) * output_dim + j] += ((cur_labels < i) & (cur_preds < i) & (cur_preds > -1)).sum()
                    _FP[(i-1) * output_dim + j] += ((cur_labels > i-1) & (cur_preds < i) & (cur_preds > -1)).sum()
                    _FN[(i-1) * output_dim + j] += ((cur_labels < i) & (cur_preds > i-1)).sum()

                    p_TP[(i-1) * output_dim + j] += ((cur_perst > i-1) & (cur_preds > i-1)).sum()
                    p_TN[(i-1) * output_dim + j] += ((cur_perst < i) & (cur_preds < i) & (cur_preds > -1)).sum()
                    p_FP[(i-1) * output_dim + j] += ((cur_perst > i-1) & (cur_preds < i) & (cur_preds > -1)).sum()
                    p_FN[(i-1) * output_dim + j] += ((cur_perst < i) & (cur_preds > i-1)).sum()

                    sim_21h_TP[(i-1) * output_dim + j] += ((cur_sim_21h > i-1) & (cur_preds > i-1)).sum()
                    sim_21h_TN[(i-1) * output_dim + j] += ((cur_sim_21h < i) & (cur_preds < i) & (cur_preds > -1)).sum()
                    sim_21h_FP[(i-1) * output_dim + j] += ((cur_sim_21h > i-1) & (cur_preds < i) & (cur_preds > -1)).sum()
                    sim_21h_FN[(i-1) * output_dim + j] += ((cur_sim_21h < i) & (cur_preds > i-1)).sum()

                    sim_avg_TP[(i-1) * output_dim + j] += ((cur_sim_avg > i-1) & (cur_preds > i-1)).sum()
                    sim_avg_TN[(i-1) * output_dim + j] += ((cur_sim_avg < i) & (cur_preds < i) & (cur_preds > -1)).sum()
                    sim_avg_FP[(i-1) * output_dim + j] += ((cur_sim_avg > i-1) & (cur_preds < i) & (cur_preds > -1)).sum()
                    sim_avg_FN[(i-1) * output_dim + j] += ((cur_sim_avg < i) & (cur_preds > i-1)).sum()

                    _RMSE_np[(i-1) * output_dim + j] += ((preds[:,j][pred_classes[:,j] > i-1] - pred_vals[:,j][pred_classes[:,j] > i-1])**2).sum().item()
                    _MAE_np[(i-1) * output_dim + j] += torch.abs(preds[:,j][pred_classes[:,j] > i-1] - pred_vals[:,j][pred_classes[:,j] > i-1]).sum().item()
                    p_RMSE_np[(i-1) * output_dim + j] += ((last_PM[:,j][pred_classes[:,j] > i-1] - pred_vals[:,j][pred_classes[:,j] > i-1])**2).sum().item()
                    p_MAE_np[(i-1) * output_dim + j] += torch.abs(last_PM[:,j][pred_classes[:,j] > i-1] - pred_vals[:,j][pred_classes[:,j] > i-1]).sum().item()
                    sim_21h_RMSE_np[(i-1) * output_dim + j] += ((sim_21h_pm_vals[:,j][pred_classes[:,j] > i-1] - pred_vals[:,j][pred_classes[:,j] > i-1])**2).sum().item()
                    sim_21h_MAE_np[(i-1) * output_dim + j] += torch.abs(sim_21h_pm_vals[:,j][pred_classes[:,j] > i-1] - pred_vals[:,j][pred_classes[:,j] > i-1]).sum().item()
                    sim_avg_RMSE_np[(i-1) * output_dim + j] += ((sim_avg_pm_vals[:,j][pred_classes[:,j] > i-1] - pred_vals[:,j][pred_classes[:,j] > i-1])**2).sum().item()
                    sim_avg_MAE_np[(i-1) * output_dim + j] += torch.abs(sim_avg_pm_vals[:,j][pred_classes[:,j] > i-1] - pred_vals[:,j][pred_classes[:,j] > i-1]).sum().item()
                    valid_count[(i-1) * output_dim + j] += (pred_classes[:,j] > i-1).sum().item()

            del preds, labels, pred_classes, pred_vals, raw_times, prev_vals, cur_labels, cur_preds, cur_perst
        # times.append(valid_times)
        # gt_vals.append(gt_vals_region[valid_time_mask])
        # gt_masks.append(region_pred_mask[valid_time_mask])
        # model_vals.append(sim_vals_region[valid_time_mask])
        # multiair_vals.append(p_vals_region[valid_time_mask])

        # os.makedirs('values_2/gt_vals', exist_ok=True)
        # os.makedirs('values_2/gt_masks', exist_ok=True)
        # os.makedirs('values_2/sim_vals', exist_ok=True)
        # os.makedirs('values_2/multiair_vals', exist_ok=True)

        # for i in range(len(times)):
        #     for j in range(len(times[i])):
        #         cur_time = times[i][j].item()
        #         np.save('values_2/gt_vals/'+str(cur_time)+'.npy', gt_vals[i][j].cpu().numpy())
        #         np.save('values_2/gt_masks/'+str(cur_time)+'.npy', gt_masks[i][j].cpu().numpy())
        #         np.save('values_2/sim_vals/'+str(cur_time)+'.npy', model_vals[i][j].cpu().numpy())
        #         np.save('values_2/multiair_vals/'+str(cur_time)+'.npy', multiair_vals[i][j].cpu().numpy())

        valid_vals_gt_np = np.array(valid_vals_gt)
        valid_vals_model_np = np.array(valid_vals_model)
        valid_vals_p_np = np.array(valid_vals_p)
        valid_vals_sim_21h = np.array(valid_vals_sim_21h)
        valid_vals_sim_avg = np.array(valid_vals_sim_avg)
        nmb_p = np.sum((valid_vals_p_np - valid_vals_gt_np)) / np.sum(valid_vals_gt_np) * 100
        nme_p = np.sum(np.abs(valid_vals_p_np - valid_vals_gt_np)) / np.sum(valid_vals_gt_np) * 100

        nmb = np.sum((valid_vals_model_np - valid_vals_gt_np)) / np.sum(valid_vals_gt_np) * 100
        nme = np.sum(np.abs(valid_vals_model_np - valid_vals_gt_np)) / np.sum(valid_vals_gt_np) * 100

        nmb_sim_21h = np.sum((valid_vals_sim_21h - valid_vals_gt_np)) / np.sum(valid_vals_gt_np) * 100
        nme_sim_21h = np.sum(np.abs(valid_vals_sim_21h - valid_vals_gt_np)) / np.sum(valid_vals_gt_np) * 100

        nmb_sim_avg = np.sum((valid_vals_sim_avg - valid_vals_gt_np)) / np.sum(valid_vals_gt_np) * 100
        nme_sim_avg = np.sum(np.abs(valid_vals_sim_avg - valid_vals_gt_np)) / np.sum(valid_vals_gt_np) * 100

        # print results of evaluation
        _RMSE = np.sqrt(_RMSE_np / valid_count)
        _MAE = _MAE_np / valid_count
        
        p_RMSE = np.sqrt(p_RMSE_np / valid_count)
        p_MAE = p_MAE_np / valid_count

        sim_21h_RMSE = np.sqrt(sim_21h_RMSE_np / valid_count)
        sim_21h_MAE = sim_21h_MAE_np / valid_count

        sim_avg_RMSE = np.sqrt(sim_avg_RMSE_np / valid_count)
        sim_avg_MAE = sim_avg_MAE_np / valid_count

        _CSI_real = (_TP / (_TP + _FN + _FP))
        _p_CSI_real = (p_TP / (p_TP + p_FN + p_FP))

        _sim_21h_CSI_real = (sim_21h_TP / (sim_21h_TP + sim_21h_FN + sim_21h_FP))
        _sim_avg_CSI_real = (sim_avg_TP / (sim_avg_TP + sim_avg_FN + sim_avg_FP))


        _F1_real = (2 * _TP / (2 * _TP + _FN + _FP))
        _p_F1_real = (2 * p_TP / (2 * p_TP + p_FN + p_FP))
        _sim_21h_F1_real = (2 * sim_21h_TP / (2 * sim_21h_TP + sim_21h_FN + sim_21h_FP))
        _sim_avg_F1_real = (2 * sim_avg_TP / (2 * sim_avg_TP + sim_avg_FN + sim_avg_FP))

        model_CSI_dict = {'normal':_CSI_real[:output_dim], 'high':_CSI_real[output_dim:2 * output_dim], 'very high':_CSI_real[2 * output_dim:]}
        persist_CSI_dict = {'normal':_p_CSI_real[:output_dim], 'high':_p_CSI_real[output_dim:2 * output_dim], 'very high':_p_CSI_real[2 * output_dim:]}
        sim_21h_CSI_dict = {'normal':_sim_21h_CSI_real[:output_dim], 'high':_sim_21h_CSI_real[output_dim:2 * output_dim], 'very high':_sim_21h_CSI_real[2 * output_dim:]}
        sim_avg_CSI_dict = {'normal':_sim_avg_CSI_real[:output_dim], 'high':_sim_avg_CSI_real[output_dim:2 * output_dim], 'very high':_sim_avg_CSI_real[2 * output_dim:]}
        model_F1_dict = {'normal':_F1_real[:output_dim], 'high':_F1_real[output_dim:2 * output_dim], 'very high':_F1_real[2 * output_dim:]}
        persist_F1_dict = {'normal':_p_F1_real[:output_dim], 'high':_p_F1_real[output_dim:2 * output_dim], 'very high':_p_F1_real[2 * output_dim:]}
        sim_21h_F1_dict = {'normal':_sim_21h_F1_real[:output_dim], 'high':_sim_21h_F1_real[output_dim:2 * output_dim], 'very high':_sim_21h_F1_real[2 * output_dim:]}
        sim_avg_F1_dict = {'normal':_sim_avg_F1_real[:output_dim], 'high':_sim_avg_F1_real[output_dim:2 * output_dim], 'very high':_sim_avg_F1_real[2 * output_dim:]}   

        model_RMSE_dict = {'normal':_RMSE[:output_dim], 'high':_RMSE[output_dim:2 * output_dim], 'very high':_RMSE[2 * output_dim:]}
        model_MAE_dict = {'normal':_MAE[:output_dim], 'high':_MAE[output_dim:2 * output_dim], 'very high':_MAE[2 * output_dim:]}
        persist_RMSE_dict = {'normal':p_RMSE[:output_dim], 'high':p_RMSE[output_dim:2 * output_dim], 'very high':p_RMSE[2 * output_dim:]}
        persist_MAE_dict = {'normal':p_MAE[:output_dim], 'high':p_MAE[output_dim:2 * output_dim], 'very high':p_MAE[2 * output_dim:]}
        sim_21h_RMSE_dict = {'normal':sim_21h_RMSE[:output_dim], 'high':sim_21h_RMSE[output_dim:2 * output_dim], 'very high':sim_21h_RMSE[2 * output_dim:]}
        sim_21h_MAE_dict = {'normal':sim_21h_MAE[:output_dim], 'high':sim_21h_MAE[output_dim:2 * output_dim], 'very high':sim_21h_MAE[2 * output_dim:]}
        sim_avg_RMSE_dict = {'normal':sim_avg_RMSE[:output_dim], 'high':sim_avg_RMSE[output_dim:2 * output_dim], 'very high':sim_avg_RMSE[2 * output_dim:]}
        sim_avg_MAE_dict = {'normal':sim_avg_MAE[:output_dim], 'high':sim_avg_MAE[output_dim:2 * output_dim], 'very high':sim_avg_MAE[2 * output_dim:]}
        per_ACC = (per_total_a1 + per_total_b2 + per_total_c3 + per_total_d4) / (per_total_a1+per_total_a2+per_total_a3+per_total_a4+per_total_b1+per_total_b2+per_total_b3+per_total_b4+per_total_c1+per_total_c2+per_total_c3+per_total_c4+per_total_d1+per_total_d2+per_total_d3+per_total_d4)
        per_POD = (per_total_c3 + per_total_c4 + per_total_d3 + per_total_d4) / (per_total_a3+per_total_a4+per_total_b3+per_total_b4+per_total_c3+per_total_c4+per_total_d3+per_total_d4)
        per_FAR = (per_total_c1 + per_total_c2 + per_total_d1 + per_total_d2) / (per_total_c1+per_total_c2+per_total_c3+per_total_c4+per_total_d1+per_total_d2+per_total_d3+per_total_d4)
        
        valid_vals_gt_np = np.array(valid_vals_gt)
        valid_vals_model_np = np.array(valid_vals_model)
        valid_vals_p_np = np.array(valid_vals_p)
        valid_vals_sim_21h_np = np.array(valid_vals_sim_21h)
        valid_vals_sim_avg_np = np.array(valid_vals_sim_avg)

        valid_vals_gt_np = valid_vals_gt_np - np.mean(valid_vals_gt_np)
        valid_vals_model_np = valid_vals_model_np - np.mean(valid_vals_model_np)
        valid_vals_p_np = valid_vals_p_np - np.mean(valid_vals_p_np)
        valid_vals_sim_21h_np = valid_vals_sim_21h_np - np.mean(valid_vals_sim_21h_np)
        valid_vals_sim_avg_np = valid_vals_sim_avg_np - np.mean(valid_vals_sim_avg_np)

        ACC = (total_a1 + total_b2 + total_c3 + total_d4) / (total_a1 + total_a2 + total_a3 + total_a4 + total_b1 + total_b2 + total_b3 + total_b4 + total_c1 + total_c2 + total_c3 + total_c4 + total_d1 + total_d2 + total_d3 + total_d4)
        POD = (total_c3 + total_c4 + total_d3 + total_d4) / (total_a3+total_a4+total_b3+total_b4+total_c3+total_c4+total_d3+total_d4)
        FAR = (total_c1 + total_c2 + total_d1 + total_d2) / (total_c1+total_c2+total_c3+total_c4+total_d1+total_d2+total_d3+total_d4)

        sim_21h_ACC = (sim_21h_total_a1 + sim_21h_total_b2 + sim_21h_total_c3 + sim_21h_total_d4) / (sim_21h_total_a1 + sim_21h_total_a2 + sim_21h_total_a3 + sim_21h_total_a4 + sim_21h_total_b1 + sim_21h_total_b2 + sim_21h_total_b3 + sim_21h_total_b4 + sim_21h_total_c1 + sim_21h_total_c2 + sim_21h_total_c3 + sim_21h_total_c4 + sim_21h_total_d1 + sim_21h_total_d2 + sim_21h_total_d3 + sim_21h_total_d4)
        sim_21h_POD = (sim_21h_total_c3 + sim_21h_total_c4 + sim_21h_total_d3 + sim_21h_total_d4) / (sim_21h_total_a3+sim_21h_total_a4+sim_21h_total_b3+sim_21h_total_b4+sim_21h_total_c3+sim_21h_total_c4+sim_21h_total_d3+sim_21h_total_d4 + 1e-9)
        sim_21h_FAR = (sim_21h_total_c1 + sim_21h_total_c2 + sim_21h_total_d1 + sim_21h_total_d2) / (sim_21h_total_c1+sim_21h_total_c2+sim_21h_total_c3+sim_21h_total_c4+sim_21h_total_d1+sim_21h_total_d2+sim_21h_total_d3+sim_21h_total_d4 + 1e-9)

        sim_avg_ACC = (sim_avg_total_a1 + sim_avg_total_b2 + sim_avg_total_c3 + sim_avg_total_d4) / (sim_avg_total_a1 + sim_avg_total_a2 + sim_avg_total_a3 + sim_avg_total_a4 + sim_avg_total_b1 + sim_avg_total_b2 + sim_avg_total_b3 + sim_avg_total_b4 + sim_avg_total_c1 + sim_avg_total_c2 + sim_avg_total_c3 + sim_avg_total_c4 + sim_avg_total_d1 + sim_avg_total_d2 + sim_avg_total_d3 + sim_avg_total_d4)
        sim_avg_POD = (sim_avg_total_c3 + sim_avg_total_c4 + sim_avg_total_d3 + sim_avg_total_d4) / (sim_avg_total_a3+sim_avg_total_a4+sim_avg_total_b3+sim_avg_total_b4+sim_avg_total_c3+sim_avg_total_c4+sim_avg_total_d3+sim_avg_total_d4 + 1e-9)
        sim_avg_FAR = (sim_avg_total_c1 + sim_avg_total_c2 + sim_avg_total_d1 + sim_avg_total_d2) / (sim_avg_total_c1+sim_avg_total_c2+sim_avg_total_c3+sim_avg_total_c4+sim_avg_total_d1+sim_avg_total_d2+sim_avg_total_d3+sim_avg_total_d4 + 1e-9)

        r = np.sum(valid_vals_model_np * valid_vals_gt_np) / (np.sqrt(np.sum(valid_vals_model_np ** 2)) * np.sqrt(np.sum(valid_vals_gt_np ** 2)))
        r_p = np.sum(valid_vals_p_np * valid_vals_gt_np) / (np.sqrt(np.sum(valid_vals_p_np ** 2)) * np.sqrt(np.sum(valid_vals_gt_np ** 2)))
        r_sim_21h = np.sum(valid_vals_sim_21h_np * valid_vals_gt_np) / (np.sqrt(np.sum(valid_vals_sim_21h_np ** 2)) * np.sqrt(np.sum(valid_vals_gt_np ** 2)))
        r_sim_avg = np.sum(valid_vals_sim_avg_np * valid_vals_gt_np) / (np.sqrt(np.sum(valid_vals_sim_avg_np ** 2)) * np.sqrt(np.sum(valid_vals_gt_np ** 2)))
        
        model_CSI_pd = pd.DataFrame(model_CSI_dict)
        persist_CSI_pd = pd.DataFrame(persist_CSI_dict)
        sim_21h_CSI_pd = pd.DataFrame(sim_21h_CSI_dict)
        sim_avg_CSI_pd = pd.DataFrame(sim_avg_CSI_dict)
        model_F1_pd = pd.DataFrame(model_F1_dict)
        persist_F1_pd = pd.DataFrame(persist_F1_dict)
        sim_21h_F1_pd = pd.DataFrame(sim_21h_F1_dict)
        sim_avg_F1_pd = pd.DataFrame(sim_avg_F1_dict)

        model_RMSE_pd = pd.DataFrame(model_RMSE_dict)
        model_MAE_pd = pd.DataFrame(model_MAE_dict)
        persist_RMSE_pd = pd.DataFrame(persist_RMSE_dict)
        persist_MAE_pd = pd.DataFrame(persist_MAE_dict)
        sim_21h_RMSE_pd = pd.DataFrame(sim_21h_RMSE_dict)
        sim_21h_MAE_pd = pd.DataFrame(sim_21h_MAE_dict)
        sim_avg_RMSE_pd = pd.DataFrame(sim_avg_RMSE_dict)
        sim_avg_MAE_pd = pd.DataFrame(sim_avg_MAE_dict)
        

        row_names = [(str(i) + "H") for i in range(1, output_dim + 1)]

        model_CSI_pd.index = row_names
        persist_CSI_pd.index = row_names   
        sim_21h_CSI_pd.index = row_names
        sim_avg_CSI_pd.index = row_names
        model_F1_pd.index = row_names
        persist_F1_pd.index = row_names
        sim_21h_F1_pd.index = row_names
        sim_avg_F1_pd.index = row_names

        model_RMSE_pd.index = row_names
        model_MAE_pd.index = row_names
        persist_RMSE_pd.index = row_names
        persist_MAE_pd.index = row_names
        sim_21h_RMSE_pd.index = row_names
        sim_21h_MAE_pd.index = row_names

        column_names = ['> 15', '> 35', '> 75']

        model_CSI_pd.columns = column_names
        persist_CSI_pd.columns = column_names
        sim_21h_CSI_pd.columns = column_names
        sim_avg_CSI_pd.columns = column_names
        model_F1_pd.columns = column_names
        persist_F1_pd.columns = column_names
        sim_21h_F1_pd.columns = column_names
        sim_avg_F1_pd.columns = column_names
        model_RMSE_pd.columns = column_names
        model_MAE_pd.columns = column_names
        persist_RMSE_pd.columns = column_names
        persist_MAE_pd.columns = column_names
        sim_21h_RMSE_pd.columns = column_names
        sim_21h_MAE_pd.columns = column_names
        sim_avg_RMSE_pd.columns = column_names
        sim_avg_MAE_pd.columns = column_names

        pd.options.display.float_format = '{:.4f}'.format

        f_log.write('persist total ACC: ' + '{:.4f}'.format(per_ACC) + '\n')
        f_log.write('persist total POD: ' + '{:.4f}'.format(per_POD) + '\n')
        f_log.write('persist total FAR: ' + '{:.4f}'.format(per_FAR) + '\n')
        f_log.write('persist total F1 score: ' + '{:.4f}'.format(2 * (per_POD * (1 - per_FAR)) / (per_POD + (1 - per_FAR))) + '\n')
        f_log.write('persist MAE: ' + '{:.4f}'.format(valid_diff_sum_p / valid_entry_cnt) + '\n')
        f_log.write('persist RMSE: ' + '{:.4f}'.format((valid_diff_squares_sum_p / valid_entry_cnt) ** 0.5) + '\n')
        f_log.write('persist NMB: ' + '{:.4f}'.format(nmb_p) + '\n')
        f_log.write('persist NME: ' + '{:.4f}'.format(nme_p) + '\n')
        f_log.write('persist R: ' + '{:.4f}'.format(r_p) + '\n')

        f_log.write('model total ACC: ' + '{:.4f}'.format(ACC) + '\n')
        f_log.write('model total POD: ' + '{:.4f}'.format(POD) + '\n')
        f_log.write('model total FAR: ' + '{:.4f}'.format(FAR) + '\n')
        f_log.write('model total F1 score: ' + '{:.4f}'.format(2 * (POD * (1 - FAR)) / (POD + (1 - FAR))) + '\n')
        f_log.write('model MAE: ' + '{:.4f}'.format(valid_diff_sum / valid_entry_cnt) + '\n')
        f_log.write('model RMSE: ' + '{:.4f}'.format((valid_diff_squares_sum / valid_entry_cnt) ** 0.5) + '\n')
        f_log.write('model NMB: ' + '{:.4f}'.format(nmb) + '\n')
        f_log.write('model NME: ' + '{:.4f}'.format(nme) + '\n')
        f_log.write('model R: ' + '{:.4f}'.format(r) + '\n')

        f_log.write('sim 21h total ACC: ' + '{:.4f}'.format(sim_21h_ACC) + '\n')
        f_log.write('sim 21h total POD: ' + '{:.4f}'.format(sim_21h_POD) + '\n')
        f_log.write('sim 21h total FAR: ' + '{:.4f}'.format(sim_21h_FAR) + '\n')
        f_log.write('sim 21h total F1 score: ' + '{:.4f}'.format(2 * (sim_21h_POD * (1 - sim_21h_FAR)) / (sim_21h_POD + (1 - sim_21h_FAR))) + '\n')
        f_log.write('sim 21h MAE: ' + '{:.4f}'.format(valid_diff_sum_sim_21h / valid_entry_cnt) + '\n')
        f_log.write('sim 21h RMSE: ' + '{:.4f}'.format((valid_diff_squares_sum_sim_21h / valid_entry_cnt) ** 0.5) + '\n')
        f_log.write('sim 21h NMB: ' + '{:.4f}'.format(nmb_sim_21h) + '\n')
        f_log.write('sim 21h NME: ' + '{:.4f}'.format(nme_sim_21h) + '\n')
        f_log.write('sim 21h R: ' + '{:.4f}'.format(r_sim_21h) + '\n')

        f_log.write('sim avg total ACC: ' + '{:.4f}'.format(sim_avg_ACC) + '\n')
        f_log.write('sim avg total POD: ' + '{:.4f}'.format(sim_avg_POD) + '\n')
        f_log.write('sim avg total FAR: ' + '{:.4f}'.format(sim_avg_FAR) + '\n')
        f_log.write('sim avg total F1 score: ' + '{:.4f}'.format(2 * (sim_avg_POD * (1 - sim_avg_FAR)) / (sim_avg_POD + (1 - sim_avg_FAR))) + '\n')
        f_log.write('sim avg MAE: ' + '{:.4f}'.format(valid_diff_sum_sim_avg / valid_entry_cnt) + '\n')
        f_log.write('sim avg RMSE: ' + '{:.4f}'.format((valid_diff_squares_sum_sim_avg / valid_entry_cnt) ** 0.5) + '\n')
        f_log.write('sim avg NMB: ' + '{:.4f}'.format(nmb_sim_avg) + '\n')
        f_log.write('sim avg NME: ' + '{:.4f}'.format(nme_sim_avg) + '\n')
        f_log.write('sim avg R: ' + '{:.4f}'.format(r_sim_avg) + '\n')

        f_log.write('persistance model CSI:\n' + persist_CSI_pd.to_string() + '\n')
        f_log.write('persistance model F1:\n' + persist_F1_pd.to_string() + '\n')
        f_log.write('persistance model RMSE:\n' + persist_RMSE_pd.to_string() + '\n')
        f_log.write('persistance model MAE:\n' + persist_MAE_pd.to_string() + '\n')
        f_log.write('MultiAir CSI:\n' + model_CSI_pd.to_string() + '\n')
        f_log.write('MultiAir F1:\n' + model_F1_pd.to_string() + '\n')
        f_log.write('MultiAir RMSE:\n' + model_RMSE_pd.to_string() + '\n')
        f_log.write('MultiAir MAE:\n' + model_MAE_pd.to_string() + '\n')
        f_log.write('simulation 21h CSI:\n' + sim_21h_CSI_pd.to_string() + '\n')
        f_log.write('simulation 21h F1:\n' + sim_21h_F1_pd.to_string() + '\n')
        f_log.write('simulation 21h RMSE:\n' + sim_21h_RMSE_pd.to_string() + '\n')
        f_log.write('simulation 21h MAE:\n' + sim_21h_MAE_pd.to_string() + '\n')
        f_log.write('simulation avg CSI:\n' + sim_avg_CSI_pd.to_string() + '\n')
        f_log.write('simulation avg F1:\n' + sim_avg_F1_pd.to_string() + '\n')
        f_log.write('simulation avg RMSE:\n' + sim_avg_RMSE_pd.to_string() + '\n')
        f_log.write('simulation avg MAE:\n' + sim_avg_MAE_pd.to_string() + '\n')

        f_log.flush()
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='evaluation MultiAir')
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed")
    parser.add_argument("--batch_size", type=int, default=24,
                        help="number of batch size")
    parser.add_argument("--data_path", type=str, default='../preprocessed_data_from_2016',
                        help="path of data")
    parser.add_argument("--sim_data_path", type=str, default='../../short_term/nier_preprocessed/CMAQ',
                        help="path of simulation data")
    parser.add_argument("--analysis_data_path", type=str, default='../analysis/CMAQ',
                        help="path of analysis data")
    parser.add_argument("--model_name", type=str, default='',
                        help="name of model to evaluate")
    parser.add_argument("--gpus", type=str, default='0',
                        help="gpu id for execution")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="hidden dimension for LSTM")
    parser.add_argument("--output_dim", type=int, default=6,
                        help="number of predictions")
    parser.add_argument("--input_dim", type=int, default=7,
                        help="input window size")
    parser.add_argument("--prev_len", type=int, default=7,
                        help="previous length for statistics of data")
    parser.add_argument("--feat_dim", type=int, default=12,
                        help="feature dimension")
    
    args = parser.parse_args()

    # set gpu devices
    if torch.cuda.is_available():
        if args.gpus == 'cpu':
            args.all_devices = None
            args.default_device = 'cpu:0'
        elif args.gpus is not None:
            args.all_devices = list(map(int, args.gpus.split(',')))
            args.default_device = args.all_devices[0]
        else:
            args.all_devices = [i for i in range(torch.cuda.device_count())]
            args.default_device = torch.cuda.current_device()
    else:
        args.all_devices = None
        args.default_device = 'cpu:0'
    torch.cuda.set_device(args.default_device)
    print(f'enabled gpu_devices: {args.all_devices}, default device: {args.default_device}')
    print(args)
    evaluation(args)
