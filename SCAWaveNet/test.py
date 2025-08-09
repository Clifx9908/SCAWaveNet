import os
import numpy as np
import torch
import scipy.io as sio
from sklearn.metrics import mean_squared_error
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
def Data_get(test_path):
     
    brcs_test, power_test, scatter_test, input_N_test, target_test = data_read(test_path)
    brcs_test, power_test, scatter_test, input_N_test, target_test = data_preprocess(brcs_test, power_test, scatter_test, input_N_test, target_test)
    
    return Data(brcs_test, power_test, scatter_test, input_N_test, target_test)


# model testing
def model_test(best_model_path, data_test, test_path, model_name):

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

    batch = 512
    best_model = torch.load(best_model_path)
    best_model.to(device)
    best_model.eval()

    test_loader = DataLoader(dataset=data_test, batch_size=batch, shuffle=False)

    with torch.no_grad():
        for batch_idx, (brcs_batch, power_batch, scatter_batch, input_N_batch, target_batch) in enumerate(tqdm(test_loader, mininterval=5), start=1):
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
        data_name = test_path.split('/')[-1].replace('.mat', '')
        model_name = model_name.replace('.pth', '')  
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

    test_path = '/file/to/path/test.mat'

    data_test_1 = Data_get(test_path)

    for i in range(1, 2):
        model_path = f'/file/to/path/model_best_rmse_exp{i}.pth'
        model_name = model_path.split('/')[-1]

        print('\n')
        print(f"**********Testing on {model_name}**********")
        print('\n')
        print("**********Testing on test.mat**********")
        print('\n')
        model_test(model_path, data_test_1, test_path, model_name)