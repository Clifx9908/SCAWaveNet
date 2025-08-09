"""
Author: Chong Zhang
Date: April 17, 2025
"""

import os
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Read .mat file
# brcs, power, scatter: [N, 4, 17, 11] (mat) -> [11, 17, 4, N] (ndarray)
# input_N: [N, 4, 9] (mat) -> [9, 4, N] (ndarray)
# target [N, 4] (mat) -> [4, N] (ndarray)
def data_read(path: str):
    with h5py.File(path, 'r') as file:
        brcs = file['brcs_downsampled'][:]
        power = file['power_analog_downsampled'][:]
        scatter = file['eff_scatter_downsampled'][:]
        input_N = file['input_N_downsampled'][:]
        target = file['ERA5_swhs_downsampled'][:]
    return brcs, power, scatter, input_N, target

# Data normalization
def Normalize(tensor):
    min_val = tensor.min(dim=0, keepdim=True)[0]
    max_val = tensor.max(dim=0, keepdim=True)[0]
    
    diff = max_val - min_val
    zero_mask = (tensor == 0)
    scaled_tensor = (tensor - min_val) / diff
    scaled_tensor[zero_mask] = 0
    return scaled_tensor

# Check for the presence of NaN or Inf values, and replace them with the mean if found
def Check(x):
    batch_size = 10000  
    num_batches = x.shape[0] // batch_size + 1

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, x.shape[0])

        batch = x[start_idx:end_idx]

        nan_flag = torch.isnan(batch).any()
        inf_flag = torch.isinf(batch).any()
        flag = nan_flag or inf_flag

        if flag:
            batch = torch.where(torch.isinf(batch), torch.full_like(batch, float('nan')), batch)
            mean_value = torch.nanmean(batch)
            batch = torch.where(torch.isnan(batch), torch.full_like(batch, mean_value.item()), batch)
            x[start_idx:end_idx] = batch
    return x

# Convert the data from NumPy to tensor type
def numpy2tensor(var):
    var = var[()]
    var = torch.from_numpy(var)
    var = var.float()
    return var

# Data preprocess function
def data_preprocess(brcs, power, scatter, input_N, target):

    brcs = numpy2tensor(brcs)
    power = numpy2tensor(power)
    scatter = numpy2tensor(scatter)
    input_N = numpy2tensor(input_N)
    
    target = numpy2tensor(target)
    target = target.unsqueeze(0)

    brcs = Check(brcs)
    power = Check(power)
    scatter = Check(scatter)
    input_N = Check(input_N)

    brcs = brcs.permute(3, 2, 0, 1)
    power = power.permute(3, 2, 0, 1)
    scatter = scatter.permute(3, 2, 0, 1)
    input_N = input_N.permute(2, 1, 0)
    target = target.permute(2, 1, 0)

    brcs = Normalize(brcs)
    power = Normalize(power)
    scatter = Normalize(scatter)
    input_N = Normalize(input_N)
    
    return brcs, power, scatter, input_N, target


# Data class
class Data(Dataset):
    def __init__(self, *inputs):
        self.inputs = inputs

    def __getitem__(self, index):
        return tuple(input_data[index] for input_data in self.inputs)

    def __len__(self):
        return len(self.inputs[0])


# Statistics for each experiment
class Statistics:
    def __init__(self, tensors=None):
        self.tensors = tensors
        self.stats = {}

    def compute_statistics(self, tensor):
        shape = tensor.shape
        max_val = torch.max(tensor)
        min_val = torch.min(tensor)
        mean_val = torch.mean(tensor)
        var_val = torch.var(tensor)
        return shape, max_val.item(), min_val.item(), mean_val.item(), var_val.item()

    def calculate_all_statistics(self):
        for tensor_name, tensor in self.tensors.items():
            self.stats[tensor_name] = self.compute_statistics(tensor)

    def save_statistics_to_pdf(self, filename="data_statistics.pdf"):
        
        self.calculate_all_statistics()

        plt.figure(figsize=(8, 6))
        plt.axis('off')
        plt.title('Tensor Statistics', fontsize=16)

        y_pos = 0.9
        for tensor_name, values in self.stats.items():
            shape, max_val, min_val, mean_val, var_val = values
            stats_text = (f"{tensor_name} Stats:\n"
                          f"Shape={shape}\n"
                          f"Max={max_val:.4f}, Min={min_val:.4f}, Mean={mean_val:.4f}, Var={var_val:.4f}")
            plt.text(0.1, y_pos, stats_text, fontsize=12, verticalalignment='top')
            y_pos -= 0.2

        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()

    def save_results_to_pdf(self, results, pdf_filename, previous_pdf_filename=None):
        items_per_page = 20
        items_per_column = 5
        page_height = 6
        y_pos_start = 1
        x_pos_start = 0

        if isinstance(results, dict):
            results = [results]

        best_rmse = min(results, key=lambda x: x['rmse'])
        best_mae = min(results, key=lambda x: abs(x['mae']))
        best_bias = min(results, key=lambda x: abs(x['bias']))
        best_cc = max(results, key=lambda x: x['cc'])
        best_mape = min(results, key=lambda x: x['mape'])

        with PdfPages(pdf_filename) as pdf:
            plt.figure(figsize=(8, page_height))
            plt.subplots_adjust(top=0.95, bottom=0.05, left=0.025, right=0.975)
            plt.axis("off")

            y_pos = y_pos_start
            x_pos = x_pos_start
            item_count = 0
            column_count = 0

            for result in results:
                epoch = result["epoch"]
                rmse = result["rmse"]
                mae = result["mae"]
                bias = result["bias"]
                cc = result["cc"]
                mape = result["mape"]

                rmse_fontweight = "bold" if result == best_rmse else "normal"
                mae_fontweight = "bold" if result == best_mae else "normal"
                bias_fontweight = "bold" if result == best_bias else "normal"
                cc_fontweight = "bold" if result == best_cc else "normal"
                mape_fontweight = "bold" if result == best_mape else "normal"

                plt.text(x_pos, y_pos, f"Epoch: {epoch}", fontsize=12, fontweight="normal")
                plt.text(x_pos, y_pos - 0.03, f"Test_RMSE: {rmse:.4f}", fontsize=10, fontweight=rmse_fontweight)
                plt.text(x_pos, y_pos - 0.06, f"Test_MAE: {mae:.4f}", fontsize=10, fontweight=mae_fontweight)
                plt.text(x_pos, y_pos - 0.09, f"Test_Bias: {bias:.4f}", fontsize=10, fontweight=bias_fontweight)
                plt.text(x_pos, y_pos - 0.12, f"Test_CC: {cc:.4f}", fontsize=10, fontweight=cc_fontweight)
                plt.text(x_pos, y_pos - 0.15, f"Test_MAPE: {mape:.4f}%", fontsize=10, fontweight=mape_fontweight)

                y_pos -= 0.22
                item_count += 1
                column_count += 1

                if column_count >= items_per_column:
                    y_pos = y_pos_start
                    x_pos += 0.24
                    column_count = 0

                if item_count >= items_per_page:
                    pdf.savefig()
                    plt.close()
                    plt.figure(figsize=(8, page_height))
                    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.025, right=0.975)
                    plt.axis("off")
                    y_pos = y_pos_start
                    x_pos = x_pos_start
                    item_count = 0
                    column_count = 0

            pdf.savefig()
            plt.close()
        print(f"Results saved to {pdf_filename}")

        if previous_pdf_filename and os.path.exists(previous_pdf_filename):
            try:
                os.remove(previous_pdf_filename)
            except PermissionError as e:
                print(f"Cannot delete {previous_pdf_filename}: {e}. Skipping deletion.")
