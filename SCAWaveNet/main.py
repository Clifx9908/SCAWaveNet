"""
Author: Chong Zhang
Date: April 17, 2025
"""

import os
import sys
import numpy as np
import torch
import scipy.io as sio
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from tqdm import tqdm
from utils import *
from model import Model


# Check GPU settings in the machine
global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("----------------------------------------")
if torch.cuda.is_available():
    print("CUDA is available")
    print("GPU: ", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")
print("Device driver: ", device)
print("----------------------------------------")


# create folders
save_dir = 'Saved_files'
os.makedirs(save_dir, exist_ok=True)
subfolders = ['models', 'outputs', 'prediction']
for subfolder in subfolders:
    os.makedirs(os.path.join(save_dir, subfolder), exist_ok=True)


# Data load and preprocess
def Data_get(train_path, valid_path, test_path):
    
    brcs_train, power_train, scatter_train, input_N_train, target_train = data_read(train_path)
    brcs_valid, power_valid, scatter_valid, input_N_valid, target_valid = data_read(valid_path)
    brcs_test, power_test, scatter_test, input_N_test, target_test = data_read(test_path)

    brcs_train, power_train, scatter_train, input_N_train, target_train = data_preprocess(brcs_train, power_train, scatter_train, input_N_train, target_train)
    brcs_valid, power_valid, scatter_valid, input_N_valid, target_valid  = data_preprocess(brcs_valid, power_valid, scatter_valid, input_N_valid, target_valid)
    brcs_test, power_test, scatter_test, input_N_test, target_test = data_preprocess(brcs_test, power_test, scatter_test, input_N_test, target_test)

    return (
        Data(brcs_train, power_train, scatter_train, input_N_train, target_train), \
        Data(brcs_valid, power_valid, scatter_valid, input_N_valid, target_valid), \
        Data(brcs_test, power_test, scatter_test, input_N_test, target_test)
    )


# Model training and validation
def Train(data_train, data_valid, max_epoch, train_batch, valid_batch, lr, weight_decay, delta, model, experiment_num, patience):
    
    train_loader = DataLoader(dataset=data_train, batch_size=train_batch, shuffle=True)
    valid_loader = DataLoader(dataset=data_valid, batch_size=valid_batch, shuffle=True)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    huber_loss = nn.HuberLoss(reduction='mean', delta=delta)
    huber_loss.to(device)

    train_losses = []
    valid_losses = [] 
    epoch_steps = []
    results = []
    statistics = Statistics()
    epoch_steps.append(len(train_loader))

    total_time_info = {
    "Total time": 0.0
    }

    best_rmse = float('inf')
    early_stopping_counter = 0

    for epoch in range(max_epoch):

        print('\n-----------Epoch: %d-----------' % (epoch+1))
        train_output = torch.zeros([0, 1]).to(device)
        train_output_1 = torch.zeros([0, 1]).to(device)
        train_output_2 = torch.zeros([0, 1]).to(device)
        train_output_3 = torch.zeros([0, 1]).to(device)
        train_output_4 = torch.zeros([0, 1]).to(device)

        train_target = torch.zeros([0, 1]).to(device)
        train_target_1 = torch.zeros([0, 1]).to(device)
        train_target_2 = torch.zeros([0, 1]).to(device)
        train_target_3 = torch.zeros([0, 1]).to(device)
        train_target_4 = torch.zeros([0, 1]).to(device)

        valid_output = torch.zeros([0, 1]).to(device)
        valid_output_1 = torch.zeros([0, 1]).to(device)
        valid_output_2 = torch.zeros([0, 1]).to(device)
        valid_output_3 = torch.zeros([0, 1]).to(device)
        valid_output_4 = torch.zeros([0, 1]).to(device)

        valid_target = torch.zeros([0, 1]).to(device)
        valid_target_1 = torch.zeros([0, 1]).to(device)
        valid_target_2 = torch.zeros([0, 1]).to(device)
        valid_target_3 = torch.zeros([0, 1]).to(device)
        valid_target_4 = torch.zeros([0, 1]).to(device)

        epoch_time_info = {
        "Total time": 0.0
        }

        print('Training...')
        model.train()
        for batch_idx, (brcs_batch, power_batch, scatter_batch, input_N_batch, target_batch) \
            in enumerate(tqdm(train_loader, mininterval=5), start=1):
            total_batches = len(train_loader)
            p_count = total_batches // 8
            
            brcs_batch = brcs_batch.to(device)
            power_batch = power_batch.to(device)
            scatter_batch = scatter_batch.to(device)
            input_N_batch = input_N_batch.to(device)
            target_batch = target_batch.to(device)
            
            output, time_info = model(brcs_batch, power_batch, scatter_batch, input_N_batch)
            output_1 = output[:, 0:1, :]
            output_2 = output[:, 1:2, :]
            output_3 = output[:, 2:3, :]
            output_4 = output[:, 3:4, :]

            target = target_batch
            target_1 = target[:, 0:1, :]  
            target_2 = target[:, 1:2, :]
            target_3 = target[:, 2:3, :]
            target_4 = target[:, 3:4, :]

            for key in epoch_time_info:
                epoch_time_info[key] += time_info[key]

            loss = huber_loss(output, target)
            train_losses.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()

            output = output.view(-1, 1)
            output_1 = output_1.view(-1, 1)
            output_2 = output_2.view(-1, 1)
            output_3 = output_3.view(-1, 1)
            output_4 = output_4.view(-1, 1)
            target = target.view(-1, 1)
            target_1 = target_1.view(-1, 1)
            target_2 = target_2.view(-1, 1)
            target_3 = target_3.view(-1, 1)
            target_4 = target_4.view(-1, 1)        
            
            train_output = torch.cat((train_output, output), 0)
            train_output_1 = torch.cat((train_output_1, output_1), 0)
            train_output_2 = torch.cat((train_output_2, output_2), 0)
            train_output_3 = torch.cat((train_output_3, output_3), 0)
            train_output_4 = torch.cat((train_output_4, output_4), 0)

            train_target = torch.cat((train_target, target), 0)
            train_target_1 = torch.cat((train_target_1, target_1), 0)
            train_target_2 = torch.cat((train_target_2, target_2), 0)
            train_target_3 = torch.cat((train_target_3, target_3), 0)
            train_target_4 = torch.cat((train_target_4, target_4), 0)

            if ((p_count > 0) and (batch_idx % p_count == 0)) or (p_count <= 0):
                sys.stdout.write("\rTrain_loss: %f" % loss)
                sys.stdout.flush()

        train_output = train_output.cpu().detach().numpy()
        train_output_1 = train_output_1.cpu().detach().numpy()
        train_output_2 = train_output_2.cpu().detach().numpy()
        train_output_3 = train_output_3.cpu().detach().numpy()
        train_output_4 = train_output_4.cpu().detach().numpy()

        train_target = train_target.cpu().detach().numpy()
        train_target_1 = train_target_1.cpu().detach().numpy()
        train_target_2 = train_target_2.cpu().detach().numpy()
        train_target_3 = train_target_3.cpu().detach().numpy()
        train_target_4 = train_target_4.cpu().detach().numpy()

        output_cc = train_output
        output_cc_1 = train_output_1
        output_cc_2 = train_output_2
        output_cc_3 = train_output_3
        output_cc_4 = train_output_4

        target_cc = train_target
        target_cc_1 = train_target_1
        target_cc_2 = train_target_2
        target_cc_3 = train_target_3
        target_cc_4 = train_target_4

        train_rmse = np.sqrt(mean_squared_error(train_output, train_target))
        train_rmse_1 = np.sqrt(mean_squared_error(train_output_1, train_target_1))
        train_rmse_2 = np.sqrt(mean_squared_error(train_output_2, train_target_2))
        train_rmse_3 = np.sqrt(mean_squared_error(train_output_3, train_target_3))
        train_rmse_4 = np.sqrt(mean_squared_error(train_output_4, train_target_4))

        train_mae = np.mean(abs(train_target - train_output))
        train_mae_1 = np.mean(abs(train_target_1 - train_output_1))
        train_mae_2 = np.mean(abs(train_target_2 - train_output_2))
        train_mae_3 = np.mean(abs(train_target_3 - train_output_3))
        train_mae_4 = np.mean(abs(train_target_4 - train_output_4))

        train_bias = np.mean(train_target - train_output)
        train_bias_1 = np.mean(train_target_1 - train_output_1)
        train_bias_2 = np.mean(train_target_2 - train_output_2)
        train_bias_3 = np.mean(train_target_3 - train_output_3)
        train_bias_4 = np.mean(train_target_4 - train_output_4)

        train_cc, _ = pearsonr(output_cc.flatten(), target_cc.flatten())
        train_cc_1, _ = pearsonr(output_cc_1.flatten(), target_cc_1.flatten())
        train_cc_2, _ = pearsonr(output_cc_2.flatten(), target_cc_2.flatten())
        train_cc_3, _ = pearsonr(output_cc_3.flatten(), target_cc_3.flatten())
        train_cc_4, _ = pearsonr(output_cc_4.flatten(), target_cc_4.flatten())
        
        train_mape = np.mean(np.abs((train_target - train_output) / train_target)) * 100.0
        train_mape_1 = np.mean(np.abs((train_target_1 - train_output_1) / train_target_1)) * 100.0
        train_mape_2 = np.mean(np.abs((train_target_2 - train_output_2) / train_target_2)) * 100.0
        train_mape_3 = np.mean(np.abs((train_target_3 - train_output_3) / train_target_3)) * 100.0
        train_mape_4 = np.mean(np.abs((train_target_4 - train_output_4) / train_target_4)) * 100.0

        print(f"\nchannel_1 training metrics\
                \nTrain_RMSE: {train_rmse_1:.4f}\
                \nTrain_MAE: {train_mae_1:.4f}\
                \nTrain_Bias: {train_bias_1:.4f}\
                \nTrain_CC: {train_cc_1:.4f}\
                \nTrain_MAPE: {train_mape_1:.4f}%")

        print(f"\nchannel_2 training metrics\
                \nTrain_RMSE: {train_rmse_2:.4f}\
                \nTrain_MAE: {train_mae_2:.4f}\
                \nTrain_Bias: {train_bias_2:.4f}\
                \nTrain_CC: {train_cc_2:.4f}\
                \nTrain_MAPE: {train_mape_2:.4f}%")

        print(f"\nchannel_3 training metrics\
                \nTrain_RMSE: {train_rmse_3:.4f}\
                \nTrain_MAE: {train_mae_3:.4f}\
                \nTrain_Bias: {train_bias_3:.4f}\
                \nTrain_CC: {train_cc_3:.4f}\
                \nTrain_MAPE: {train_mape_3:.4f}%")

        print(f"\nchannel_4 training metrics\
                \nTrain_RMSE: {train_rmse_4:.4f}\
                \nTrain_MAE: {train_mae_4:.4f}\
                \nTrain_Bias: {train_bias_4:.4f}\
                \nTrain_CC: {train_cc_4:.4f}\
                \nTrain_MAPE: {train_mape_4:.4f}%")

        print(f"\nall_channel training metrics\
                \nTrain_RMSE: {train_rmse:.4f}\
                \nTrain_MAE: {train_mae:.4f}\
                \nTrain_Bias: {train_bias:.4f}\
                \nTrain_CC: {train_cc:.4f}\
                \nTrain_MAPE: {train_mape:.4f}%")

        print('\nValidation...')
        model.eval()
        with torch.no_grad():
            for batch_idx, (brcs_batch1, power_batch1, scatter_batch1, input_N_batch1, target_batch1) \
                in enumerate(tqdm(valid_loader, mininterval=5), start=1):
                total_batches = len(valid_loader)
                p_count = total_batches // 2

                brcs_batch1 = brcs_batch1.to(device)
                power_batch1 = power_batch1.to(device)
                scatter_batch1 = scatter_batch1.to(device)
                input_N_batch1 = input_N_batch1.to(device)
                target_batch1 = target_batch1.to(device)

                output1, time_info1 = model(brcs_batch1, power_batch1, scatter_batch1, input_N_batch1)
                output1_1 = output1[:, 0:1, :]
                output1_2 = output1[:, 1:2, :]
                output1_3 = output1[:, 2:3, :]
                output1_4 = output1[:, 3:4, :]

                target1 = target_batch1
                target1_1 = target1[:, 0:1, :]  
                target1_2 = target1[:, 1:2, :]
                target1_3 = target1[:, 2:3, :]
                target1_4 = target1[:, 3:4, :]

                for key in epoch_time_info:
                    epoch_time_info[key] += time_info1[key]

                valloss = huber_loss(output1, target1)
                valid_losses.append(valloss.item())     
                
                output1 = output1.view(-1, 1)
                output1_1 = output1_1.view(-1, 1)
                output1_2 = output1_2.view(-1, 1)
                output1_3 = output1_3.view(-1, 1)
                output1_4 = output1_4.view(-1, 1)
                target1 = target1.view(-1, 1)
                target1_1 = target1_1.view(-1, 1)
                target1_2 = target1_2.view(-1, 1)
                target1_3 = target1_3.view(-1, 1)
                target1_4 = target1_4.view(-1, 1)

                valid_output = torch.cat((valid_output, output1), 0)
                valid_output_1 = torch.cat((valid_output_1, output1_1), 0)
                valid_output_2 = torch.cat((valid_output_2, output1_2), 0)
                valid_output_3 = torch.cat((valid_output_3, output1_3), 0)
                valid_output_4 = torch.cat((valid_output_4, output1_4), 0)

                valid_target = torch.cat((valid_target, target1), 0)
                valid_target_1 = torch.cat((valid_target_1, target1_1), 0)
                valid_target_2 = torch.cat((valid_target_2, target1_2), 0)
                valid_target_3 = torch.cat((valid_target_3, target1_3), 0)
                valid_target_4 = torch.cat((valid_target_4, target1_4), 0)
                
                if ((p_count > 0) and (batch_idx % p_count == 0)) or (p_count <= 0):
                    sys.stdout.write("\rValid_loss: %f " % valloss)
                    sys.stdout.flush()

        valid_output = valid_output.cpu().detach().numpy()
        valid_output_1 = valid_output_1.cpu().detach().numpy()
        valid_output_2 = valid_output_2.cpu().detach().numpy()
        valid_output_3 = valid_output_3.cpu().detach().numpy()
        valid_output_4 = valid_output_4.cpu().detach().numpy()

        valid_target = valid_target.cpu().detach().numpy()
        valid_target_1 = valid_target_1.cpu().detach().numpy()
        valid_target_2 = valid_target_2.cpu().detach().numpy()
        valid_target_3 = valid_target_3.cpu().detach().numpy()
        valid_target_4 = valid_target_4.cpu().detach().numpy()

        output_cc = valid_output
        output_cc_1 = valid_output_1
        output_cc_2 = valid_output_2
        output_cc_3 = valid_output_3
        output_cc_4 = valid_output_4

        target_cc = valid_target
        target_cc_1 = valid_target_1
        target_cc_2 = valid_target_2
        target_cc_3 = valid_target_3
        target_cc_4 = valid_target_4

        valid_rmse = np.sqrt(mean_squared_error(valid_output, valid_target))
        valid_rmse_1 = np.sqrt(mean_squared_error(valid_output_1, valid_target_1))
        valid_rmse_2 = np.sqrt(mean_squared_error(valid_output_2, valid_target_2))
        valid_rmse_3 = np.sqrt(mean_squared_error(valid_output_3, valid_target_3))
        valid_rmse_4 = np.sqrt(mean_squared_error(valid_output_4, valid_target_4))

        valid_mae = np.mean(abs(valid_target - valid_output))
        valid_mae_1 = np.mean(abs(valid_target_1 - valid_output_1))
        valid_mae_2 = np.mean(abs(valid_target_2 - valid_output_2))
        valid_mae_3 = np.mean(abs(valid_target_3 - valid_output_3))
        valid_mae_4 = np.mean(abs(valid_target_4 - valid_output_4))

        valid_bias = np.mean(valid_target - valid_output)
        valid_bias_1 = np.mean(valid_target_1 - valid_output_1)
        valid_bias_2 = np.mean(valid_target_2 - valid_output_2)
        valid_bias_3 = np.mean(valid_target_3 - valid_output_3)
        valid_bias_4 = np.mean(valid_target_4 - valid_output_4)

        valid_cc, _ = pearsonr(output_cc.flatten(), target_cc.flatten())
        valid_cc_1, _ = pearsonr(output_cc_1.flatten(), target_cc_1.flatten())
        valid_cc_2, _ = pearsonr(output_cc_2.flatten(), target_cc_2.flatten())
        valid_cc_3, _ = pearsonr(output_cc_3.flatten(), target_cc_3.flatten())
        valid_cc_4, _ = pearsonr(output_cc_4.flatten(), target_cc_4.flatten())
        
        valid_mape = np.mean(np.abs((valid_target - valid_output) / valid_target)) * 100.0
        valid_mape_1 = np.mean(np.abs((valid_target_1 - valid_output_1) / valid_target_1)) * 100.0
        valid_mape_2 = np.mean(np.abs((valid_target_2 - valid_output_2) / valid_target_2)) * 100.0
        valid_mape_3 = np.mean(np.abs((valid_target_3 - valid_output_3) / valid_target_3)) * 100.0
        valid_mape_4 = np.mean(np.abs((valid_target_4 - valid_output_4) / valid_target_4)) * 100.0

        print(f"\nchannel_1 valid metrics\
                \nValid_RMSE: {valid_rmse_1:.4f}\
                \nValid_MAE: {valid_mae_1:.4f}\
                \nValid_Bias: {valid_bias_1:.4f}\
                \nValid_CC: {valid_cc_1:.4f}\
                \nValid_MAPE: {valid_mape_1:.4f}%")

        print(f"\nchannel_2 valid metrics\
                \nValid_RMSE: {valid_rmse_2:.4f}\
                \nValid_MAE: {valid_mae_2:.4f}\
                \nValid_Bias: {valid_bias_2:.4f}\
                \nValid_CC: {valid_cc_2:.4f}\
                \nValid_MAPE: {valid_mape_2:.4f}%")

        print(f"\nchannel_3 valid metrics\
                \nValid_RMSE: {valid_rmse_3:.4f}\
                \nValid_MAE: {valid_mae_3:.4f}\
                \nValid_Bias: {valid_bias_3:.4f}\
                \nValid_CC: {valid_cc_3:.4f}\
                \nValid_MAPE: {valid_mape_3:.4f}%")

        print(f"\nchannel_4 valid metrics\
                \nValid_RMSE: {valid_rmse_4:.4f}\
                \nValid_MAE: {valid_mae_4:.4f}\
                \nValid_Bias: {valid_bias_4:.4f}\
                \nValid_CC: {valid_cc_4:.4f}\
                \nValid_MAPE: {valid_mape_4:.4f}%")

        print(f"\nall_channel valid metrics\
                \nValid_RMSE: {valid_rmse:.4f}\
                \nValid_MAE: {valid_mae:.4f}\
                \nValid_Bias: {valid_bias:.4f}\
                \nValid_CC: {valid_cc:.4f}\
                \nValid_MAPE: {valid_mape:.4f}%")

        epoch_result = {
            "epoch": epoch + 1,
            "rmse": valid_rmse,
            "mae": valid_mae,
            "bias": valid_bias,
            "cc": valid_cc,
            "mape": valid_mape
        }
        results.append(epoch_result)

        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            early_stopping_counter = 0
            torch.save(model, f'./Saved_files/models/model_best_rmse_exp{experiment_num}.pth')
        else:
            early_stopping_counter += 1
        
        best_rmse = min(result['rmse'] for result in results)
        best_rmse_index = next(i for i, result in enumerate(results) if result['rmse'] == best_rmse)
        print(f"Best RMSE: {best_rmse:.4f} at epoch {best_rmse_index + 1}")

        if early_stopping_counter >= patience:
            print(f"Early stopping after {patience} epochs without improvement.")
            break

        if epoch > 0:
            previous_pdf = f"./Saved_files/outputs/valid_results_epoch{epoch}_exp{experiment_num}.pdf"
        else:
            previous_pdf = None
        
        current_pdf = f"./Saved_files/outputs/valid_results_epoch{epoch + 1}_exp{experiment_num}.pdf"
        statistics.save_results_to_pdf(results, current_pdf, previous_pdf)

        for key in total_time_info:
            total_time_info[key] += epoch_time_info[key]

    print("\n========== Final Accumulated Time Info ==========")
    for key, value in total_time_info.items():
        print(f"{key}: {value:.4f}s")
    
    best_model_path = f'./Saved_files/models/model_best_rmse_exp{experiment_num}.pth'
    best_rmse_epoch = best_rmse_index + 1

    return best_model_path, best_rmse, best_rmse_epoch


def run_experiments(data_train, data_valid, data_test):

    num_experiments = 3
    for experiment_num in range(1, num_experiments + 1):
        max_epoch = 75
        train_batch = 512
        valid_batch = 512
        test_batch = 512
        lr = 1.4e-4
        weight_decay = 1e-2
        delta = 2.0
        patience = 15

        model = Model()
        model.to(device)

        best_model_path, best_rmse, best_rmse_epoch = Train(data_train, data_valid, max_epoch, train_batch, valid_batch, lr, weight_decay, delta,
                                                            model, experiment_num, patience)
        print(f"Best RMSE for experiment {experiment_num}: {best_rmse:.4f} at epoch {best_rmse_epoch}")

        print("\nTesting the best model...")
        test_output = torch.zeros([0, 1]).to(device)
        test_output_1 = torch.zeros([0, 1]).to(device)
        test_output_2 = torch.zeros([0, 1]).to(device)
        test_output_3 = torch.zeros([0, 1]).to(device)
        test_output_4 = torch.zeros([0, 1]).to(device)

        test_target = torch.zeros([0, 1]).to(device)
        test_target_1 = torch.zeros([0, 1]).to(device)
        test_target_2 = torch.zeros([0, 1]).to(device)
        test_target_3 = torch.zeros([0, 1]).to(device)
        test_target_4 = torch.zeros([0, 1]).to(device)
        
        best_model = torch.load(best_model_path)
        best_model.to(device)
        best_model.eval()
        
        test_loader = DataLoader(dataset=data_test, batch_size=test_batch, shuffle=False)

        with torch.no_grad():
            for batch_idx, (brcs_batch, power_batch, scatter_batch, input_N_batch, target_batch) in enumerate(tqdm(test_loader, mininterval=5), start=1):
                total_batches = len(test_loader)
                p_count = total_batches // 2
                brcs_batch = brcs_batch.to(device)
                power_batch = power_batch.to(device)
                scatter_batch = scatter_batch.to(device)
                input_N_batch = input_N_batch.to(device)
                target_batch = target_batch.to(device)

                output, time_info = best_model(brcs_batch, power_batch, scatter_batch, input_N_batch)
                output_1 = output[:, 0:1, :]
                output_2 = output[:, 1:2, :]
                output_3 = output[:, 2:3, :]
                output_4 = output[:, 3:4, :]

                target = target_batch
                target_1 = target[:, 0:1, :]  
                target_2 = target[:, 1:2, :]
                target_3 = target[:, 2:3, :]
                target_4 = target[:, 3:4, :]

                output = output.view(-1, 1)
                output_1 = output_1.view(-1, 1)
                output_2 = output_2.view(-1, 1)
                output_3 = output_3.view(-1, 1)
                output_4 = output_4.view(-1, 1)
                target = target.view(-1, 1)
                target_1 = target_1.view(-1, 1)
                target_2 = target_2.view(-1, 1)
                target_3 = target_3.view(-1, 1)
                target_4 = target_4.view(-1, 1)           

                test_output = torch.cat((test_output, output), 0)
                test_output_1 = torch.cat((test_output_1, output_1), 0)
                test_output_2 = torch.cat((test_output_2, output_2), 0)
                test_output_3 = torch.cat((test_output_3, output_3), 0)
                test_output_4 = torch.cat((test_output_4, output_4), 0)

                test_target = torch.cat((test_target, target), 0)
                test_target_1 = torch.cat((test_target_1, target_1), 0)
                test_target_2 = torch.cat((test_target_2, target_2), 0)
                test_target_3 = torch.cat((test_target_3, target_3), 0)
                test_target_4 = torch.cat((test_target_4, target_4), 0)

            test_output = test_output.cpu().detach().numpy()
            test_output_1 = test_output_1.cpu().detach().numpy()
            test_output_2 = test_output_2.cpu().detach().numpy()
            test_output_3 = test_output_3.cpu().detach().numpy()
            test_output_4 = test_output_4.cpu().detach().numpy()

            test_target = test_target.cpu().detach().numpy()
            test_target_1 = test_target_1.cpu().detach().numpy()
            test_target_2 = test_target_2.cpu().detach().numpy()
            test_target_3 = test_target_3.cpu().detach().numpy()
            test_target_4 = test_target_4.cpu().detach().numpy()

            output_cc = test_output
            output_cc_1 = test_output_1
            output_cc_2 = test_output_2
            output_cc_3 = test_output_3
            output_cc_4 = test_output_4

            target_cc = test_target
            target_cc_1 = test_target_1
            target_cc_2 = test_target_2
            target_cc_3 = test_target_3
            target_cc_4 = test_target_4

            test_rmse = np.sqrt(mean_squared_error(test_output, test_target))
            test_rmse_1 = np.sqrt(mean_squared_error(test_output_1, test_target_1))
            test_rmse_2 = np.sqrt(mean_squared_error(test_output_2, test_target_2))
            test_rmse_3 = np.sqrt(mean_squared_error(test_output_3, test_target_3))
            test_rmse_4 = np.sqrt(mean_squared_error(test_output_4, test_target_4))

            test_mae = np.mean(abs(test_target - test_output))
            test_mae_1 = np.mean(abs(test_target_1 - test_output_1))
            test_mae_2 = np.mean(abs(test_target_2 - test_output_2))
            test_mae_3 = np.mean(abs(test_target_3 - test_output_3))
            test_mae_4 = np.mean(abs(test_target_4 - test_output_4))

            test_bias = np.mean(test_target - test_output)
            test_bias_1 = np.mean(test_target_1 - test_output_1)
            test_bias_2 = np.mean(test_target_2 - test_output_2)
            test_bias_3 = np.mean(test_target_3 - test_output_3)
            test_bias_4 = np.mean(test_target_4 - test_output_4)

            test_cc, _ = pearsonr(output_cc.flatten(), target_cc.flatten())
            test_cc_1, _ = pearsonr(output_cc_1.flatten(), target_cc_1.flatten())
            test_cc_2, _ = pearsonr(output_cc_2.flatten(), target_cc_2.flatten())
            test_cc_3, _ = pearsonr(output_cc_3.flatten(), target_cc_3.flatten())
            test_cc_4, _ = pearsonr(output_cc_4.flatten(), target_cc_4.flatten())
            
            test_mape = np.mean(np.abs((test_target - test_output) / test_target)) * 100.0
            test_mape_1 = np.mean(np.abs((test_target_1 - test_output_1) / test_target_1)) * 100.0
            test_mape_2 = np.mean(np.abs((test_target_2 - test_output_2) / test_target_2)) * 100.0
            test_mape_3 = np.mean(np.abs((test_target_3 - test_output_3) / test_target_3)) * 100.0
            test_mape_4 = np.mean(np.abs((test_target_4 - test_output_4) / test_target_4)) * 100.0

            print(f"\nchannel_1 testing metrics\
                    \nTest_RMSE: {test_rmse_1:.4f}\
                    \nTest_MAE: {test_mae_1:.4f}\
                    \nTest_Bias: {test_bias_1:.4f}\
                    \nTest_CC: {test_cc_1:.4f}\
                    \nTest_MAPE: {test_mape_1:.4f}%")

            print(f"\nchannel_2 testing metrics\
                    \nTest_RMSE: {test_rmse_2:.4f}\
                    \nTest_MAE: {test_mae_2:.4f}\
                    \nTest_Bias: {test_bias_2:.4f}\
                    \nTest_CC: {test_cc_2:.4f}\
                    \nTest_MAPE: {test_mape_2:.4f}%")

            print(f"\nchannel_3 testing metrics\
                    \nTest_RMSE: {test_rmse_3:.4f}\
                    \nTest_MAE: {test_mae_3:.4f}\
                    \nTest_Bias: {test_bias_3:.4f}\
                    \nTest_CC: {test_cc_3:.4f}\
                    \nTest_MAPE: {test_mape_3:.4f}%")

            print(f"\nchannel_4 testing metrics\
                    \nTest_RMSE: {test_rmse_4:.4f}\
                    \nTest_MAE: {test_mae_4:.4f}\
                    \nTest_Bias: {test_bias_4:.4f}\
                    \nTest_CC: {test_cc_4:.4f}\
                    \nTest_MAPE: {test_mape_4:.4f}%")

            print(f"\nall_channel testing metrics\
                    \nTest_RMSE: {test_rmse:.4f}\
                    \nTest_MAE: {test_mae:.4f}\
                    \nTest_Bias: {test_bias:.4f}\
                    \nTest_CC: {test_cc:.4f}\
                    \nTest_MAPE: {test_mape:.4f}%")

            # save file after testing
            data_name = test_path.split('/')[-1]
            data_name = data_name.replace('.mat', '')
            model_name = best_model_path.split('/')[-1].replace('.pth', '')  
            mat_save_path = os.path.join(save_dir, 'prediction', f'{model_name}_{data_name}_predictions.mat')
            
            mat_data = {
                'pred_channel_1': test_output_1,
                'pred_channel_2': test_output_2,
                'pred_channel_3': test_output_3,
                'pred_channel_4': test_output_4,
                'true_channel_1': test_target_1,
                'true_channel_2': test_target_2,
                'true_channel_3': test_target_3,
                'true_channel_4': test_target_4
            }
            sio.savemat(mat_save_path, mat_data)
            
            print(f"\nPredictions and ground truth saved to: {mat_save_path}")


if __name__ == '__main__':

    # dataset path
    train_path = '/file/to/path/train.mat'
    valid_path = '/file/to/path/valid.mat'
    test_path = '/file/to/path/test.mat'

    print('Getting Data...')

    data_train, data_valid, data_test = Data_get(train_path, valid_path, test_path)

    run_experiments(data_train, data_valid, data_test)