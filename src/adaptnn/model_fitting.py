from adaptnn.retina_datasets import NiruDataset, MultiContrastFullFieldJN05Dataset
from adaptnn.artificial_datasets import LinearNonlinear
from adaptnn.visual_neuron_convnets import PopulationConvNet, PopulationFullFieldNet, penalize_convnet_weights
import torch
import numpy as np

class ArtificialModel:
    def __init__(self, dataset_params = {"filter_spatial" : (15,15),
                                         "filter_time" : 10,
                                         "num_cells" : 4,
                                         "out_noise_std_train" : None,
                                         "filter_rank" : 2},
                       net_params = {"layer_time_lengths" : (10,),
                                     "layer_rf_pixel_widths" : (15,),
                                     "layer_channels" : (4,),
                                     "layer_spatio_temporal_factorization_type" : ('spatial',),
                                     "out_normalization" : True,
                                     "layer_normalization" : True}):

        self.dataset = LinearNonlinear(**dataset_params)

        if(self.dataset.temporal_only):
            self.model_class = PopulationFullFieldNet
            if("frame_width" in net_params):
                del(net_params["frame_width"] )
            if("frame_height" in net_params):
                del(net_params["frame_height"] )
            if("layer_rf_pixel_widths" in net_params):
                del(net_params["layer_rf_pixel_widths"] )
        else:
            self.model_class = PopulationConvNet
            net_params["frame_width"]  = self.dataset.frame_width
            net_params["frame_height"] = self.dataset.frame_height

        self.model = self.model_class(num_cells=self.dataset.num_cells,
                                       **net_params)
        
        self.dataset.time_padding_bins = self.model.time_padding


    def predict(self):
        '''
        Returns:
            Y_test_predicted : Tensor of size (Time,Cells)
            Y_test_true : Tensor of size (Time,Cells)
        '''
        with torch.no_grad():
            X_test, Y_test = self.dataset.get_test()
            return self.model(X_test), Y_test
        

    def get_loss_function(self, **kwargs):
        return torch.nn.MSELoss( **kwargs)
    
    def get_penalty_function(self,  en_lambda = 0.01,
                                    en_alpha = 0.5,
                                    fl_lambda_x = 0.01,
                                    fl_lambda_y = 0.01,
                                    fl_lambda_t = 0.01,
                                    lin_en_lambda = 0.01,
                                    lin_en_alpha = 0.5):
        return lambda mm : penalize_convnet_weights(mm, en_lambda, en_alpha,
                                                        fl_lambda_x, fl_lambda_y, fl_lambda_t,
                                                        lin_en_lambda, lin_en_alpha)
    
    def train(self,
              epochs : int,
              optimizer_params = {"lr" : 1e-4},
              scheduler_params = {"start_factor" : 1.0, "end_factor" : 0.1, "total_iters" : 2000},
              batch_params = {"batch_size":16, "shuffle":True},
              penalty_params = {},
              loss_params = {},
              print_every=50):

        optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_params)
        if(scheduler_params is not None):
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_params)
        else:
            scheduler = None

        penalty = self.get_penalty_function(**penalty_params)
        criterion = self.get_loss_function(**loss_params)

        train_dataloader = torch.utils.data.DataLoader(self.dataset,
                                                       generator=torch.Generator(device=self.dataset.X_train.device), 
                                                       **batch_params)

        for epoch in range(epochs):
            running_loss = 0
            for X_t, Y_t in train_dataloader:
        
                #convert numpy array to torch Variable
                optimizer.zero_grad()
                
                #Forward to get outputs
                outputs=self.model(X_t)
                
                #calculate loss
                loss=criterion(outputs, Y_t) + penalty(self.model)
                
                #getting gradients wrt parameters
                loss.backward()
                
                #updating parameters
                optimizer.step()
                if(scheduler is not None):
                    scheduler.step()

                running_loss += loss.item()
                
            if(np.isnan(running_loss) or np.isinf(running_loss)):
                raise RuntimeError(f"Loss function returning invalid results: {running_loss}")
            if((epoch+1) % print_every == 0):
                print(f"epoch {epoch+1}, loss {running_loss}, step size {optimizer.param_groups[0]['lr']}")

class NiruDataModel:
    def __init__(self, dataset_params = {"recording" : "A-noise", "dtype" : torch.float32, "segment_length_bins" : 300},
                       net_params = {
                           "layer_channels" : (8,8),
                           "layer_rf_pixel_widths" : (15,11),
                           "layer_time_lengths" : (40,12),
                           "layer_spatio_temporal_factorization_type" : ("spatial", "spatial"),
                           "layer_spatio_temporal_rank" : [6,6],
                           "hidden_activation" : torch.nn.Softplus,
                           "layer_normalization" : (False,False),
                           "out_normalization" : True,
                           "out_activation" : torch.nn.Softplus}):

        self.dataset = NiruDataset(**dataset_params)

        self.model = PopulationConvNet(num_cells=self.dataset.num_cells,
                                       frame_width=self.dataset.frame_width,
                                       frame_height=self.dataset.frame_height,
                                       **net_params)
        
        self.dataset.time_padding_bins = self.model.time_padding


    def predict(self):
        with torch.no_grad():
            X_test, Y_test = self.dataset.get_test()
            return self.model(X_test), Y_test
        

    def get_loss_function(self, **kwargs):
        return torch.nn.PoissonNLLLoss(log_input=(not self.model.nonlinear_output), **kwargs)
    
    def get_penalty_function(self,  en_lambda = 0.01,
                                    en_alpha = 0.5,
                                    fl_lambda_x = 0.01,
                                    fl_lambda_y = 0.01,
                                    fl_lambda_t = 0.01,
                                    lin_en_lambda = 0.01,
                                    lin_en_alpha = 0.5):
        return lambda mm : penalize_convnet_weights(mm, en_lambda, en_alpha,
                                                        fl_lambda_x, fl_lambda_y, fl_lambda_t,
                                                        lin_en_lambda, lin_en_alpha)
    
    def train(self,
              epochs : int,
              optimizer_params = {"lr" : 1e-4},
              scheduler_params = {"start_factor" : 1.0, "end_factor" : 0.1, "total_iters" : 2000},
              batch_params = {"batch_size":16, "shuffle":True},
              penalty_params = {},
              loss_params = {},
              print_every=50,
              print_every_batch=None):

        optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_params)
        if(scheduler_params is not None):
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_params)
        else:
            scheduler = None

        penalty = self.get_penalty_function(**penalty_params)
        criterion = self.get_loss_function(**loss_params)

        train_dataloader = torch.utils.data.DataLoader(self.dataset,
                                                       generator=torch.Generator(device=self.dataset.X_train.device), 
                                                       **batch_params)

        for epoch in range(epochs):
            running_loss = 0
            for batch_num,batch_data in enumerate(train_dataloader):
                X_t, Y_t = batch_data
                if(print_every_batch is not None and (batch_num+1) % print_every_batch == 0):
                    print(f"\tbatch {batch_num+1}")
        
                #convert numpy array to torch Variable
                optimizer.zero_grad()
                
                #Forward to get outputs
                outputs=self.model(X_t)
                
                #calculate loss
                loss=criterion(outputs, Y_t) + penalty(self.model)
                
                #getting gradients wrt parameters
                loss.backward()
                
                #updating parameters
                optimizer.step()
                if(scheduler is not None):
                    scheduler.step()

                running_loss += loss.item()
                
            if(np.isnan(running_loss) or np.isinf(running_loss)):
                raise RuntimeError(f"Loss function returning invalid results: {running_loss}")
            if(print_every is not None and (epoch+1) % print_every == 0):
                print(f"epoch {epoch+1}, loss {running_loss}, step size {optimizer.param_groups[0]['lr']}")




class MCJN05DataModel:
    def __init__(self, dataset_params = {"dtype" : torch.float32},
                       net_params = {"out_activation" : torch.nn.Softplus}):

        self.dataset = MultiContrastFullFieldJN05Dataset(**dataset_params)

        self.model = PopulationFullFieldNet(num_cells=self.dataset.num_cells,
                                       **net_params)
        
        self.dataset.time_padding_bins = self.model.time_padding


    def predict_long(self):
        with torch.no_grad():
            X_test, Y_test = self.dataset.get_test_long()
            return self.model(X_test), Y_test
    def predict_rpt(self, contrast):
        with torch.no_grad():
            X_test, Y_test = self.dataset.get_test_rpt(contrast)
            return self.model(X_test), Y_test
        

    def get_loss_function(self, **kwargs):
        return torch.nn.PoissonNLLLoss(log_input=(not self.model.nonlinear_output), **kwargs)
    
    def get_penalty_function(self,  en_lambda = 0.01,
                                    en_alpha = 0.5,
                                    fl_lambda_t = 0.01,
                                    lin_en_lambda = 0.01,
                                    lin_en_alpha = 0.5):
        return lambda mm : penalize_convnet_weights(mm, en_lambda, en_alpha,
                                                        None, None, fl_lambda_t,
                                                        lin_en_lambda, lin_en_alpha)
    
    def train(self,
              epochs : int,
              optimizer_params = {"lr" : 1e-4},
              scheduler_params = None,#{"start_factor" : 1.0, "end_factor" : 0.1, "total_iters" : 2000},
              batch_params = {"batch_size":8, "shuffle":True},
              penalty_params = {},
              loss_params = {},
              print_every=10):

        optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_params)
        if(scheduler_params is not None):
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_params)
        else:
            scheduler = None

        penalty = self.get_penalty_function(**penalty_params)
        criterion = self.get_loss_function(**loss_params)

        
        train_dataloader = torch.utils.data.DataLoader(self.dataset,
                                                       generator=torch.Generator(device=self.dataset.X_train.device), 
                                                       **batch_params)

        for epoch in range(epochs):
            running_loss = 0
            for X_t, Y_t in train_dataloader:
        
                #convert numpy array to torch Variable
                optimizer.zero_grad()
                
                #Forward to get outputs
                outputs=self.model(X_t)
                
                #calculate loss
                loss=criterion(outputs, Y_t) + penalty(self.model)
                
                #getting gradients wrt parameters
                loss.backward()
                
                #updating parameters
                optimizer.step()
                if(scheduler is not None):
                    scheduler.step()

                running_loss += loss.item()
                
            if(np.isnan(running_loss) or np.isinf(running_loss)):
                raise RuntimeError(f"Loss function returning invalid results: {running_loss}")
            if((epoch+1) % print_every == 0):
                print(f"epoch {epoch+1}, loss {running_loss}, step size {optimizer.param_groups[0]['lr']}")